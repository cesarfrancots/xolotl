use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall { pub id: String, pub name: String, pub args: String } // args = raw JSON text from the model

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage { pub prompt: u64, pub completion: u64, pub cached: u64 }

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    System(String),
    User(String),
    Assistant { text: String, reasoning: Option<String>, tool_calls: Vec<ToolCall> },
    Tool { call_id: String, content: String },
}

impl Message {
    /// OpenAI-style chat message. An empty assistant `text` is sent as `""` (ruling R1, unconfirmed by the API probe).
    pub fn to_wire(&self) -> Value {
        match self {
            Message::System(s) => json!({"role": "system", "content": s}),
            Message::User(s) => json!({"role": "user", "content": s}),
            Message::Assistant { text, reasoning, tool_calls } => {
                let mut w = json!({"role": "assistant", "content": text});
                if let Some(r) = reasoning { w["reasoning_content"] = json!(r); }
                if !tool_calls.is_empty() {
                    w["tool_calls"] = tool_calls.iter().map(|c| json!({
                        "id": c.id, "type": "function",
                        "function": {"name": c.name, "arguments": c.args},
                    })).collect();
                }
                w
            }
            Message::Tool { call_id, content } => json!({"role": "tool", "tool_call_id": call_id, "content": content}),
        }
    }

    /// Characters of model-visible content (the spec's `chars / 3` size estimate input).
    pub fn chars(&self) -> usize {
        match self {
            Message::System(s) | Message::User(s) => s.chars().count(),
            Message::Assistant { text, reasoning, tool_calls } => {
                text.chars().count()
                    + reasoning.as_deref().map_or(0, |r| r.chars().count())
                    + tool_calls.iter().map(|c| c.name.chars().count() + c.args.chars().count()).sum::<usize>()
            }
            Message::Tool { content, .. } => content.chars().count(),
        }
    }
}

pub enum Delta<'a> { Text(&'a str), Reasoning(&'a str), Notice(String) }

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reply {
    pub text: String, pub reasoning: String, pub tool_calls: Vec<ToolCall>,
    pub usage: Usage, pub finish_reason: String, pub interrupted: bool,
}

#[derive(Debug, PartialEq)]
pub enum LlmError { Auth(String), ContextOverflow(String), Http { status: u16, body: String }, Network(String) }

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LlmError::Auth(m) => write!(f, "authentication failed: {m}"),
            LlmError::ContextOverflow(m) => write!(f, "context window exceeded: {m}"),
            LlmError::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            LlmError::Network(m) => write!(f, "network error: {m}"),
        }
    }
}

pub trait Model {
    fn complete(&mut self, messages: &[Message], tools: &[Value],
                on: &mut dyn FnMut(Delta), cancel: &AtomicBool) -> Result<Reply, LlmError>;
}

/// Replays canned replies in order and records each request as `(messages, tool count)`.
pub struct Scripted { replies: VecDeque<Result<Reply, LlmError>>, pub requests: Vec<(Vec<Message>, usize)> }

impl Scripted {
    pub fn new(replies: Vec<Result<Reply, LlmError>>) -> Self {
        Scripted { replies: replies.into(), requests: Vec::new() }
    }
}

impl Model for Scripted {
    fn complete(&mut self, messages: &[Message], tools: &[Value],
                on: &mut dyn FnMut(Delta), _cancel: &AtomicBool) -> Result<Reply, LlmError> {
        self.requests.push((messages.to_vec(), tools.len()));
        let reply = self.replies.pop_front().unwrap_or_else(|| Err(LlmError::Network("script exhausted".into())))?;
        if !reply.reasoning.is_empty() { on(Delta::Reasoning(&reply.reasoning)); }
        if !reply.text.is_empty() { on(Delta::Text(&reply.text)); }
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn assistant_wire_shape() {
        let m = Message::Assistant { text: String::new(), reasoning: Some("r".into()),
            tool_calls: vec![ToolCall { id: "c1".into(), name: "read".into(), args: r#"{"path":"a"}"#.into() }] };
        let w = m.to_wire();
        assert_eq!(w["role"], "assistant");
        assert_eq!(w["content"], "");                 // or null, if M0 probe (d) says so
        assert_eq!(w["tool_calls"][0]["id"], "c1");
        assert_eq!(w["tool_calls"][0]["type"], "function");
        assert_eq!(w["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(w["tool_calls"][0]["function"]["arguments"], r#"{"path":"a"}"#);
        assert_eq!(w["reasoning_content"], "r");
    }
    #[test] fn assistant_without_tools_or_reasoning_omits_fields() {
        let w = Message::Assistant { text: "hi".into(), reasoning: None, tool_calls: vec![] }.to_wire();
        assert!(w.get("tool_calls").is_none() && w.get("reasoning_content").is_none());
    }
    #[test] fn tool_and_user_wire_shape() {
        let w = Message::Tool { call_id: "c1".into(), content: "x".into() }.to_wire();
        assert_eq!((w["role"].as_str(), w["tool_call_id"].as_str(), w["content"].as_str()), (Some("tool"), Some("c1"), Some("x")));
        assert_eq!(Message::User("u".into()).to_wire(), serde_json::json!({"role":"user","content":"u"}));
    }
    #[test] fn chars_counts_all_content() {
        let m = Message::Assistant { text: "ab".into(), reasoning: Some("cd".into()),
            tool_calls: vec![ToolCall { id: "x".into(), name: "read".into(), args: "{}".into() }] };
        assert_eq!(m.chars(), 2 + 2 + 4 + 2); // text + reasoning + name + args
    }
    #[test] fn scripted_replays_in_order_and_records() {
        let ok = |t: &str| Ok(Reply { text: t.into(), ..Default::default() });
        let mut m = Scripted::new(vec![ok("a"), ok("b")]);
        let cancel = AtomicBool::new(false);
        let mut seen = String::new();
        let r = m.complete(&[Message::User("hi".into())], &[], &mut |d| if let Delta::Text(t) = d { seen.push_str(t) }, &cancel).unwrap();
        assert_eq!((r.text.as_str(), seen.as_str()), ("a", "a"));
        assert_eq!(m.complete(&[], &[serde_json::json!({})], &mut |_| {}, &cancel).unwrap().text, "b");
        assert!(matches!(m.complete(&[], &[], &mut |_| {}, &cancel), Err(LlmError::Network(_))));
        assert_eq!(m.requests.len(), 3);
        assert_eq!(m.requests[0], (vec![Message::User("hi".into())], 0));
        assert_eq!(m.requests[1].1, 1);
    }
}

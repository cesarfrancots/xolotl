use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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

pub struct ZaiConfig {
    pub base_url: String, pub model: String, pub api_key: String,
    pub temperature: f64, pub top_p: f64, pub reasoning_effort: Option<String>,
}

pub struct ZaiClient { cfg: ZaiConfig, agent: ureq::Agent }

impl ZaiClient {
    pub fn new(cfg: ZaiConfig) -> Self {
        // No global/body timeout: a stream may legitimately run for minutes.
        let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder()
            .http_status_as_error(false).timeout_connect(Some(Duration::from_secs(30))).build());
        ZaiClient { cfg, agent }
    }

    /// One HTTP attempt. On error, also returns the parsed `Retry-After` seconds.
    fn attempt(&self, body: &str, state: &mut StreamState, on: &mut dyn FnMut(Delta),
               cancel: &AtomicBool) -> Result<Reply, (LlmError, Option<u64>)> {
        let url = format!("{}/chat/completions", self.cfg.base_url.trim_end_matches('/'));
        let mut resp = self.agent.post(&url)
            .header("Authorization", format!("Bearer {}", self.cfg.api_key))
            .content_type("application/json")
            .send(body)
            .map_err(|e| (LlmError::Network(e.to_string()), None))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let retry_after = resp.headers().get("retry-after")
                .and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok());
            let text = resp.body_mut().read_to_string().unwrap_or_default();
            return Err((classify(status, &text), retry_after));
        }
        let mut reader = BufReader::new(resp.body_mut().as_reader());
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => return Err((LlmError::Network(e.to_string()), None)),
            }
            match sse_data(&line) {
                Some("[DONE]") => break,
                // ponytail: unparseable data lines are skipped; a mid-stream error object is not surfaced until the probe shows its shape.
                Some(data) => if let Ok(chunk) = serde_json::from_str::<Value>(data) { apply_chunk(state, &chunk, on) },
                None => {}
            }
            if cancel.load(Ordering::Relaxed) { state.reply.interrupted = true; break; }
        }
        Ok(finish(std::mem::take(state)))
    }
}

impl Model for ZaiClient {
    fn complete(&mut self, messages: &[Message], tools: &[Value],
                on: &mut dyn FnMut(Delta), cancel: &AtomicBool) -> Result<Reply, LlmError> {
        let body = request_body(&self.cfg, messages, tools).to_string();
        let mut n = 0;
        loop {
            let mut state = StreamState::default();
            let (err, retry_after) = match self.attempt(&body, &mut state, on, cancel) {
                Ok(reply) => return Ok(reply),
                Err(e) => e,
            };
            let Some(delay) = retry_delay(&err, n, state.emitted, retry_after) else { return Err(err) };
            on(Delta::Notice(format!("{err}; retrying in {}s", delay.as_secs())));
            if !wait(delay, cancel) { return Ok(Reply { interrupted: true, ..Default::default() }); }
            n += 1;
        }
    }
}

fn request_body(cfg: &ZaiConfig, messages: &[Message], tools: &[Value]) -> Value {
    let mut b = json!({
        "model": cfg.model, "messages": messages.iter().map(Message::to_wire).collect::<Vec<_>>(),
        "stream": true, "temperature": cfg.temperature, "top_p": cfg.top_p,
    });
    if !tools.is_empty() { b["tools"] = json!(tools); }
    if let Some(e) = &cfg.reasoning_effort { b["reasoning_effort"] = json!(e); }
    b
}

/// Sleeps in 100 ms slices; returns false as soon as `cancel` is set.
fn wait(d: Duration, cancel: &AtomicBool) -> bool {
    let end = Instant::now() + d;
    while !cancel.load(Ordering::Relaxed) {
        let left = end.saturating_duration_since(Instant::now());
        if left.is_zero() { return true; }
        std::thread::sleep(left.min(Duration::from_millis(100)));
    }
    false
}

/// Payload of an SSE `data:` line; None for blank lines, `:` comments and other fields.
fn sse_data(line: &str) -> Option<&str> {
    line.trim_end().strip_prefix("data:").map(str::trim_start)
}

#[derive(Default)]
pub struct StreamState { pub reply: Reply, pub emitted: bool, calls: BTreeMap<u64, ToolCall> }

pub fn apply_chunk(state: &mut StreamState, chunk: &Value, on: &mut dyn FnMut(Delta)) {
    if let Some(u) = chunk["usage"].as_object() {
        let n = |v: Option<&Value>| v.and_then(Value::as_u64).unwrap_or(0);
        state.reply.usage = Usage { prompt: n(u.get("prompt_tokens")), completion: n(u.get("completion_tokens")),
            cached: n(u.get("prompt_tokens_details").and_then(|d| d.get("cached_tokens"))) };
    }
    let choice = &chunk["choices"][0];
    let delta = &choice["delta"];
    if let Some(r) = delta["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
        state.reply.reasoning.push_str(r);
        state.emitted = true;
        on(Delta::Reasoning(r));
    }
    if let Some(t) = delta["content"].as_str().filter(|s| !s.is_empty()) {
        state.reply.text.push_str(t);
        state.emitted = true;
        on(Delta::Text(t));
    }
    for tc in delta["tool_calls"].as_array().into_iter().flatten() {
        // ponytail: a fragment without `index` merges into call 0; revisit if the API probe shows providers omit it.
        let call = state.calls.entry(tc["index"].as_u64().unwrap_or(0))
            .or_insert_with(|| ToolCall { id: String::new(), name: String::new(), args: String::new() });
        if call.id.is_empty() { call.id.push_str(tc["id"].as_str().unwrap_or("")); }
        if call.name.is_empty() { call.name.push_str(tc["function"]["name"].as_str().unwrap_or("")); }
        call.args.push_str(tc["function"]["arguments"].as_str().unwrap_or(""));
    }
    if let Some(f) = choice["finish_reason"].as_str() { state.reply.finish_reason = f.to_string(); }
}

pub fn finish(state: StreamState) -> Reply {
    Reply { tool_calls: state.calls.into_values().collect(), ..state.reply }
}

/// Substrings that mark a 400 as a context overflow. Provisional until the API probe records the real body.
const OVERFLOW_MARKERS: [&str; 5] = ["context", "too long", "max length", "exceed", "1261"];

pub fn classify(status: u16, body: &str) -> LlmError {
    let lower = body.to_lowercase();
    match status {
        401 | 403 => LlmError::Auth(format!("HTTP {status}: {body}")),
        400 if OVERFLOW_MARKERS.iter().any(|m| lower.contains(m)) => LlmError::ContextOverflow(body.to_string()),
        _ => LlmError::Http { status, body: body.to_string() },
    }
}

/// Backoff before retry number `attempt + 1` (0-based `attempt`): `Retry-After` capped at 60 s, else 1/2/4/8 s.
pub fn retry_delay(err: &LlmError, attempt: u32, emitted: bool, retry_after_s: Option<u64>) -> Option<Duration> {
    let retryable = match err {
        LlmError::Http { status, .. } => *status == 429 || *status >= 500,
        LlmError::Network(_) => true,
        _ => false,
    };
    if !retryable || emitted || attempt >= 4 { return None; }
    Some(Duration::from_secs(retry_after_s.map_or(1 << attempt, |s| s.min(60))))
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

    // Provisional until the API probe (Task 0) records the real overflow body in docs/glm-api-notes.md.
    const OVERFLOW_BODY_FROM_NOTES: &str = r#"{"error":{"code":"1261","message":"Prompt exceeds max length"}}"#;

    /// Every `data:` payload up to `[DONE]`; blank lines and `:` comments are skipped.
    fn sse_chunks(raw: &str) -> Vec<Value> {
        raw.lines().filter_map(sse_data).take_while(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap()).collect()
    }
    fn run(chunks: &[Value]) -> (StreamState, Vec<String>) {
        let (mut st, mut seen) = (StreamState::default(), Vec::new());
        for c in chunks {
            apply_chunk(&mut st, c, &mut |d| seen.push(match d {
                Delta::Text(t) => format!("T:{t}"), Delta::Reasoning(r) => format!("R:{r}"), Delta::Notice(n) => format!("N:{n}"),
            }));
        }
        (st, seen)
    }
    fn call(id: &str, name: &str, args: &str) -> ToolCall { ToolCall { id: id.into(), name: name.into(), args: args.into() } }

    #[test] fn assembles_split_tool_call_deltas() {
        let cs = [json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read","arguments":""}}]}}]}),
                  json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}),
                  json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a\"}"}}]}}]}),
                  json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}],
                         "usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}})];
        let r = finish(run(&cs).0);
        assert_eq!(r.tool_calls, vec![call("c1", "read", r#"{"path":"a"}"#)]);
        assert_eq!(r.usage, Usage { prompt: 10, completion: 5, cached: 4 });
        assert_eq!(r.finish_reason, "tool_calls");
    }
    #[test] fn whole_tool_call_in_one_chunk() {
        let c = json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"c1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},"finish_reason":"tool_calls"}]});
        let r = finish(run(&[c]).0);
        assert_eq!(r.tool_calls, vec![call("c1", "read", r#"{"path":"a"}"#)]);
        assert_eq!(r.finish_reason, "tool_calls");
    }
    #[test] fn two_tool_calls_keep_index_order() {
        let cs = [json!({"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"write","arguments":"{\"x\":"}}]}}]}),
                  json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{}"}}]}}]}),
                  json!({"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"1}"}}]}}]})];
        let r = finish(run(&cs).0);
        assert_eq!(r.tool_calls, vec![call("c1", "read", "{}"), call("c2", "write", r#"{"x":1}"#)]);
    }
    #[test] fn text_and_reasoning_stream_separately() {
        let cs = sse_chunks(concat!(
            ": keepalive\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data:{\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\r\n\r\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"after done\"}}]}\n\n"));
        assert_eq!(cs.len(), 3);
        let (st, seen) = run(&cs);
        assert_eq!(seen, ["R:think", "T:Hel", "T:lo"]);
        assert!(st.emitted);
        let r = finish(st);
        assert_eq!((r.text.as_str(), r.reasoning.as_str(), r.finish_reason.as_str()), ("Hello", "think", "stop"));
    }
    #[test] fn malformed_or_empty_chunks_are_ignored() {
        let cs = [json!({}), json!({"choices":[]}), json!({"choices":[{}]}), json!({"usage":null}),
                  json!({"choices":[{"delta":{"content":"","reasoning_content":null,"tool_calls":"x"},"finish_reason":null}]}),
                  json!({"choices":[{"delta":{"tool_calls":[{}]}}]})];
        let (st, seen) = run(&cs);
        assert!(seen.is_empty() && !st.emitted);
        let r = finish(st);
        assert_eq!((r.text.as_str(), r.usage, r.finish_reason.as_str()), ("", Usage::default(), ""));
    }
    #[test] fn classify_errors() {
        assert!(matches!(classify(401, ""), LlmError::Auth(_)));
        assert!(matches!(classify(403, ""), LlmError::Auth(_)));
        assert!(matches!(classify(400, OVERFLOW_BODY_FROM_NOTES), LlmError::ContextOverflow(_)));
        assert!(matches!(classify(400, "bad request"), LlmError::Http { status: 400, .. }));
        assert!(matches!(classify(429, ""), LlmError::Http { status: 429, .. }));
        assert!(matches!(classify(500, OVERFLOW_BODY_FROM_NOTES), LlmError::Http { status: 500, .. })); // overflow is 400 only
    }
    #[test] fn retry_policy() {
        let rl = LlmError::Http { status: 429, body: String::new() };
        let s = Duration::from_secs;
        assert_eq!(retry_delay(&rl, 0, false, None), Some(s(1)));
        assert_eq!(retry_delay(&rl, 3, false, None), Some(s(8)));
        assert_eq!(retry_delay(&rl, 4, false, None), None);             // 5 attempts total
        assert_eq!(retry_delay(&rl, 0, false, Some(7)), Some(s(7)));    // Retry-After honored
        assert_eq!(retry_delay(&rl, 0, false, Some(600)), Some(s(60))); // capped at 60s
        assert_eq!(retry_delay(&LlmError::Http { status: 503, body: String::new() }, 1, false, None), Some(s(2)));
        assert_eq!(retry_delay(&LlmError::Network("x".into()), 0, false, None), Some(s(1)));
        assert_eq!(retry_delay(&LlmError::Network("x".into()), 0, true, None), None); // already streamed: never retry
        assert_eq!(retry_delay(&LlmError::Auth("x".into()), 0, false, None), None);
        assert_eq!(retry_delay(&LlmError::Http { status: 400, body: String::new() }, 0, false, None), None);
    }
    #[test] fn request_body_fields() {
        let mut cfg = ZaiConfig { base_url: "u".into(), model: "glm-5.3-flash".into(), api_key: "k".into(),
                                  temperature: 1.0, top_p: 0.95, reasoning_effort: None };
        let msgs = [Message::User("hi".into())];
        let b = request_body(&cfg, &msgs, &[]);
        assert_eq!((b["model"].as_str(), b["stream"].as_bool(), b["temperature"].as_f64(), b["top_p"].as_f64()),
                   (Some("glm-5.3-flash"), Some(true), Some(1.0), Some(0.95)));
        assert_eq!(b["messages"], json!([{"role":"user","content":"hi"}]));
        assert!(b.get("tools").is_none() && b.get("reasoning_effort").is_none() && b.get("stream_options").is_none());
        cfg.reasoning_effort = Some("max".into());
        let b = request_body(&cfg, &msgs, &[json!({"type":"function"})]);
        assert_eq!((b["tools"].as_array().map(Vec::len), b["reasoning_effort"].as_str()), (Some(1), Some("max")));
    }
    #[test] fn backoff_sleep_honors_cancel() {
        assert!(wait(Duration::ZERO, &AtomicBool::new(false)));
        let t = std::time::Instant::now();
        assert!(!wait(Duration::from_secs(30), &AtomicBool::new(true)));
        assert!(t.elapsed() < Duration::from_secs(1));
    }
}

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
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
    /// OpenAI-style chat message. An empty assistant `text` is sent as `""`; the API accepts `""`, null or no `content` there.
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
    /// The client's thinking cap ended the call (`interrupted` stays for Ctrl+C). Only ever with reasoning and no text or tool calls.
    pub capped: bool,
    /// Milliseconds from the first reasoning token to the first content or tool-call token (or to the end of the call
    /// if none came; 0 without reasoning). `None`: the model does not measure it.
    pub think_ms: Option<u64>,
    /// The `reasoning_effort` the request carried; `None`: none was sent, so the API default (`max`) applied.
    pub effort: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum LlmError { Auth(String), ContextOverflow(String), Quota(String), Http { status: u16, body: String }, Network(String) }

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LlmError::Auth(m) => write!(f, "authentication failed: {m}"),
            LlmError::ContextOverflow(m) => write!(f, "context window exceeded: {m}"),
            LlmError::Quota(m) => write!(f, "quota or balance exhausted: {m}"),
            LlmError::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            LlmError::Network(m) => write!(f, "network error: {m}"),
        }
    }
}

/// The `reasoning_effort` values the API documents, strongest first. It silently accepts any other value
/// (docs/glm-api-notes.md #5).
pub const EFFORTS: [&str; 7] = ["max", "xhigh", "high", "medium", "low", "minimal", "none"];

/// The weaker of `base` (`None`: the API default, `max`) and `cap`; a cap that isn't a documented value changes nothing.
pub fn lower_effort<'a>(base: Option<&'a str>, cap: Option<&'a str>) -> Option<&'a str> {
    let rank = |e: &str| EFFORTS.iter().position(|x| *x == e);
    match (base, cap) {
        (_, None) => base,
        (None, Some(c)) => rank(c).map(|_| c),
        (Some(b), Some(c)) => match (rank(b), rank(c)) {
            (Some(rb), Some(rc)) if rc > rb => Some(c),
            _ => Some(b),
        },
    }
}

pub trait Model {
    fn complete(&mut self, messages: &[Message], tools: &[Value],
                on: &mut dyn FnMut(Delta), cancel: &AtomicBool) -> Result<Reply, LlmError>;

    /// Limits the reasoning effort of the calls that follow (never raises it); `None` lifts the limit. Ignored by default.
    fn cap_effort(&mut self, _cap: Option<&str>) {}

    /// Ends a call that reasons for this long before its first content or tool-call token, with `Reply::capped` and the
    /// partial reasoning; applies to the calls that follow, `None` lifts it. Ignored by default.
    fn think_cap(&mut self, _cap: Option<Duration>) {}
}

/// Replays canned replies in order and records each request as `(messages, tool count)`, and each effort and thinking cap it was given.
#[allow(dead_code)] // test double: only the tests use it
pub struct Scripted {
    replies: VecDeque<Result<Reply, LlmError>>, pub requests: Vec<(Vec<Message>, usize)>,
    pub caps: Vec<Option<String>>, pub think_caps: Vec<Option<Duration>>,
}

#[allow(dead_code)] // test double: only the tests use it
impl Scripted {
    pub fn new(replies: Vec<Result<Reply, LlmError>>) -> Self {
        Scripted { replies: replies.into(), requests: Vec::new(), caps: Vec::new(), think_caps: Vec::new() }
    }
}

impl Model for Scripted {
    fn cap_effort(&mut self, cap: Option<&str>) { self.caps.push(cap.map(String::from)); }

    fn think_cap(&mut self, cap: Option<Duration>) { self.think_caps.push(cap); }

    fn complete(&mut self, messages: &[Message], tools: &[Value],
                on: &mut dyn FnMut(Delta), _cancel: &AtomicBool) -> Result<Reply, LlmError> {
        self.requests.push((messages.to_vec(), tools.len()));
        let mut reply = self.replies.pop_front().unwrap_or_else(|| Err(LlmError::Network("script exhausted".into())))?;
        // like a client with no configured effort: the last cap it was given is what the request carried
        reply.effort = lower_effort(None, self.caps.last().and_then(|c| c.as_deref())).map(String::from);
        if !reply.reasoning.is_empty() { on(Delta::Reasoning(&reply.reasoning)); }
        if !reply.text.is_empty() { on(Delta::Text(&reply.text)); }
        Ok(reply)
    }
}

pub struct ZaiConfig {
    pub base_url: String, pub model: String, pub api_key: String,
    pub temperature: f64, pub top_p: f64, pub reasoning_effort: Option<String>,
}

/// No message from the worker for this long means the server went silent.
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
/// How often `attempt` wakes to check `cancel` while waiting for the worker.
const POLL: Duration = Duration::from_millis(100);

pub struct ZaiClient { cfg: ZaiConfig, agent: ureq::Agent, idle: Duration, effort_cap: Option<String>, think_cap: Option<Duration> }

/// What the worker thread reports back to `attempt`.
enum Msg {
    Open(u16),                       // 2xx headers received
    Line(String),                    // one raw line of the body
    Eof,                             // body ended
    Failed(LlmError, Option<u64>),   // transport error or non-2xx, plus parsed Retry-After seconds
}

impl ZaiClient {
    pub fn new(cfg: ZaiConfig) -> Self {
        // No global/body timeout: a stream may legitimately run for minutes; `attempt` enforces the idle limit.
        let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder()
            .http_status_as_error(false).timeout_connect(Some(Duration::from_secs(30))).build());
        ZaiClient { cfg, agent, idle: IDLE_TIMEOUT, effort_cap: None, think_cap: None }
    }

    /// The configured reasoning effort, lowered to the current cap.
    fn effort(&self) -> Option<&str> {
        lower_effort(self.cfg.reasoning_effort.as_deref(), self.effort_cap.as_deref())
    }

    /// The request body, with the reasoning effort lowered to the current cap.
    fn body(&self, messages: &[Message], tools: &[Value]) -> Value {
        let mut b = request_body(&self.cfg, messages, tools);
        if let Some(e) = self.effort() { b["reasoning_effort"] = json!(e); }
        b
    }

    /// Starts the request on a worker thread. A blocking read cannot be interrupted, so the worker simply
    /// outlives a cancelled attempt: it exits when its send fails (receiver dropped) or the socket closes.
    fn open(&self, body: &str) -> mpsc::Receiver<Msg> {
        let (tx, rx) = mpsc::channel();
        let agent = self.agent.clone();
        let url = format!("{}/chat/completions", self.cfg.base_url.trim_end_matches('/'));
        let auth = format!("Bearer {}", self.cfg.api_key);
        let body = body.to_string();
        std::thread::spawn(move || {
            if let Err((e, retry_after)) = read_stream(&agent, &url, &auth, &body, &tx) {
                let _ = tx.send(Msg::Failed(e, retry_after));
            }
        });
        rx
    }

    /// One HTTP attempt. On error, also returns the parsed `Retry-After` seconds.
    fn attempt(&self, body: &str, state: &mut StreamState, on: &mut dyn FnMut(Delta),
               cancel: &AtomicBool) -> Result<Reply, (LlmError, Option<u64>)> {
        let rx = self.open(body);
        let (mut status, mut done, mut last) = (0, false, Instant::now());
        let mut think = Think::default();
        loop {
            if cancel.load(Ordering::Relaxed) {
                state.reply.interrupted = true;
                state.reply.think_ms = Some(think.ms());
                return Ok(finish(std::mem::take(state)));
            }
            // Reasoning only so far, for the whole cap: end the call like a cancel does (the worker is simply abandoned) and keep
            // what was thought. Never once content or a tool call has begun, and never before any reasoning (a silent server is
            // the idle limit's business, and a capped reply always has reasoning to keep).
            if let (Some(cap), Some(start), None) = (self.think_cap, think.start, think.end)
                && start.elapsed() >= cap
            {
                state.reply.capped = true;
                state.reply.think_ms = Some(think.ms());
                return Ok(finish(std::mem::take(state)));
            }
            let msg = match rx.recv_timeout(POLL) {
                Ok(m) => m,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if last.elapsed() >= self.idle { return Err((LlmError::Network("stream stalled".into()), None)); }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => Msg::Eof,
            };
            last = Instant::now();
            match msg {
                Msg::Open(s) => status = s,
                Msg::Failed(e, retry_after) => return Err((e, retry_after)),
                Msg::Eof => break,
                Msg::Line(line) => {
                    if let Some(e) = stream_error(status, &line) { return Err((e, None)); }
                    match sse_data(&line) {
                        Some("[DONE]") => { done = true; break; }
                        // ponytail: unparseable data lines are skipped.
                        Some(data) => if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                            apply_chunk(state, &chunk, on);
                            think.seen(state);
                        },
                        None => {}
                    }
                }
            }
        }
        if !stream_complete(done, &state.reply.finish_reason) {
            return Err((LlmError::Network("stream ended before [DONE]".into()), None));
        }
        state.reply.think_ms = Some(think.ms());
        Ok(finish(std::mem::take(state)))
    }
}

/// When a call's thinking started (first reasoning token) and ended (first content or tool-call token).
#[derive(Default)]
struct Think { start: Option<Instant>, end: Option<Instant> }

impl Think {
    /// Called after every applied chunk.
    fn seen(&mut self, st: &StreamState) {
        if self.start.is_none() && !st.reply.reasoning.is_empty() { self.start = Some(Instant::now()); }
        if self.end.is_none() && (!st.reply.text.is_empty() || !st.calls.is_empty()) { self.end = Some(Instant::now()); }
    }

    /// Milliseconds spent thinking so far: until the output began, or until now if it has not.
    fn ms(&self) -> u64 {
        self.start.map_or(0, |s| self.end.unwrap_or_else(Instant::now).saturating_duration_since(s).as_millis() as u64)
    }
}

/// Worker body: send the request, classify non-2xx, then forward the body line by line.
fn read_stream(agent: &ureq::Agent, url: &str, auth: &str, body: &str,
               tx: &mpsc::Sender<Msg>) -> Result<(), (LlmError, Option<u64>)> {
    let mut resp = agent.post(url)
        .header("Authorization", auth)
        .header("Accept-Encoding", "identity") // compression would buffer deltas
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
    if tx.send(Msg::Open(status)).is_err() { return Ok(()); }
    let mut reader = BufReader::new(resp.body_mut().as_reader());
    let mut line = String::new();
    loop {
        match reader.read_line(&mut line) {
            Ok(0) => { let _ = tx.send(Msg::Eof); return Ok(()); }
            Ok(_) => if tx.send(Msg::Line(std::mem::take(&mut line))).is_err() { return Ok(()); },
            Err(e) => return Err((LlmError::Network(e.to_string()), None)),
        }
    }
}

/// A stream is complete only if it ended with `[DONE]` or carried a `finish_reason`; otherwise it was cut off.
fn stream_complete(done: bool, finish_reason: &str) -> bool { done || !finish_reason.is_empty() }

impl Model for ZaiClient {
    fn complete(&mut self, messages: &[Message], tools: &[Value],
                on: &mut dyn FnMut(Delta), cancel: &AtomicBool) -> Result<Reply, LlmError> {
        let body = self.body(messages, tools).to_string();
        let effort = self.effort().map(String::from);
        let mut n = 0;
        loop {
            let mut state = StreamState::default();
            let (err, retry_after) = match self.attempt(&body, &mut state, on, cancel) {
                Ok(reply) => return Ok(Reply { effort, ..reply }),
                Err(e) => e,
            };
            let Some(delay) = retry_delay(&err, n, state.emitted, retry_after) else { return Err(err) };
            on(Delta::Notice(format!("{err}; retrying in {}s", delay.as_secs())));
            if !wait(delay, cancel) { return Ok(Reply { interrupted: true, effort, ..Default::default() }); }
            n += 1;
        }
    }

    fn cap_effort(&mut self, cap: Option<&str>) { self.effort_cap = cap.map(String::from); }

    fn think_cap(&mut self, cap: Option<Duration>) { self.think_cap = cap; }
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
        let (id, name, args) = (tc["id"].as_str().unwrap_or(""), tc["function"]["name"].as_str().unwrap_or(""),
                                tc["function"]["arguments"].as_str().unwrap_or(""));
        if id.is_empty() && name.is_empty() && args.is_empty() { continue; } // nothing to add, and no phantom call
        // ponytail: Z.ai always sends `index` (docs/glm-api-notes.md #3); a fragment without one merges into call 0.
        let call = state.calls.entry(tc["index"].as_u64().unwrap_or(0))
            .or_insert_with(|| ToolCall { id: String::new(), name: String::new(), args: String::new() });
        if call.id.is_empty() { call.id.push_str(id); }
        if call.name.is_empty() { call.name.push_str(name); }
        call.args.push_str(args);
    }
    if let Some(f) = choice["finish_reason"].as_str() { state.reply.finish_reason = f.to_string(); }
}

pub fn finish(state: StreamState) -> Reply {
    Reply { tool_calls: state.calls.into_values().collect(), ..state.reply }
}

/// The only HTTP 429 business codes worth retrying: rate limit (1302) and service overloaded (1305). Every other
/// code on a 429 is permanent (no balance, plan expired, period limit, fair use, ...; docs/glm-api-notes.md "Retry caution").
const TRANSIENT_429_CODES: [&str; 2] = ["1302", "1305"];

/// `error.code` of a Z.ai error envelope. The API sends a string; a number is accepted too.
fn error_code(v: &Value) -> Option<String> {
    match &v["error"]["code"] {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Context overflow (docs/glm-api-notes.md #6): `error.code` 1261, else `error.message` (or the raw body when it
/// is not JSON) containing "prompt too long". Looser wording such as "context" or "exceed" is deliberately not matched.
fn overflows(body: &str) -> bool {
    let too_long = |s: &str| s.to_lowercase().contains("prompt too long");
    match serde_json::from_str::<Value>(body) {
        Ok(v) => error_code(&v).as_deref() == Some("1261") || v["error"]["message"].as_str().is_some_and(too_long),
        Err(_) => too_long(body),
    }
}

/// The message of a permanent 429 body: it has an `error.code` that is not in `TRANSIENT_429_CODES`.
/// None for transient codes and for bodies with no parseable code.
fn quota_message(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    let code = error_code(&v)?;
    (!TRANSIENT_429_CODES.contains(&code.as_str())).then(|| v["error"]["message"].as_str().unwrap_or(body).to_string())
}

pub fn classify(status: u16, body: &str) -> LlmError {
    let http = || LlmError::Http { status, body: body.to_string() };
    match status {
        401 | 403 => LlmError::Auth(format!("HTTP {status}: {body}")),
        400 if overflows(body) => LlmError::ContextOverflow(body.to_string()),
        429 => quota_message(body).map_or_else(http, LlmError::Quota),
        _ => http(),
    }
}

/// An error reported inside a 2xx stream: a `data:` payload or bare line that is JSON with a top-level `error`.
fn stream_error(status: u16, line: &str) -> Option<LlmError> {
    let line = line.trim();
    let payload = sse_data(line).unwrap_or(line);
    if serde_json::from_str::<Value>(payload).ok()?["error"].is_null() { return None; }
    Some(if overflows(payload) { LlmError::ContextOverflow(payload.to_string()) }
         else { LlmError::Http { status, body: payload.to_string() } })
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
        assert_eq!(w["content"], "");                 // the API also accepts null or an omitted content here
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

    // docs/glm-api-notes.md #6: HTTP 400, code 1261, message "Prompt too long", inside the live-verified `{"error":{code,message}}`
    // envelope. The overflow itself was never reproduced live, so this body is built from the docs row, not captured.
    const OVERFLOW_BODY_FROM_NOTES: &str = r#"{"error":{"code":"1261","message":"Prompt too long"}}"#;
    // docs/glm-api-notes.md "Error format": live-captured HTTP 429 with no balance on the pay-as-you-go API.
    const QUOTA_BODY_FROM_NOTES: &str = r#"{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}"#;

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
                  json!({"choices":[{"delta":{"tool_calls":[{}]}}]}),
                  json!({"choices":[{"delta":{"tool_calls":[{"index":3,"type":"function","function":{}}]}}]})];
        let (st, seen) = run(&cs);
        assert!(seen.is_empty() && !st.emitted);
        let r = finish(st);
        assert_eq!((r.text.as_str(), r.usage, r.finish_reason.as_str()), ("", Usage::default(), ""));
        assert!(r.tool_calls.is_empty()); // fragments with no id, name or arguments leave no phantom call
    }
    #[test] fn classify_errors() {
        assert!(matches!(classify(401, ""), LlmError::Auth(_)));
        assert!(matches!(classify(403, ""), LlmError::Auth(_)));
        assert!(matches!(classify(400, OVERFLOW_BODY_FROM_NOTES), LlmError::ContextOverflow(_)));
        assert!(matches!(classify(400, "bad request"), LlmError::Http { status: 400, .. }));
        assert!(matches!(classify(429, ""), LlmError::Http { status: 429, .. }));
        assert!(matches!(classify(500, OVERFLOW_BODY_FROM_NOTES), LlmError::Http { status: 500, .. })); // overflow is 400 only
    }
    #[test] fn overflow_rule() {
        // code 1261 wins whatever the message says, as a string or a number
        assert!(matches!(classify(400, r#"{"error":{"code":"1261","message":"anything"}}"#), LlmError::ContextOverflow(_)));
        assert!(matches!(classify(400, r#"{"error":{"code":1261,"message":"anything"}}"#), LlmError::ContextOverflow(_)));
        // fallback: the message, case-insensitively, or the raw body when it is not JSON
        assert!(matches!(classify(400, r#"{"error":{"code":"9999","message":"The PROMPT TOO LONG for this model"}}"#), LlmError::ContextOverflow(_)));
        assert!(matches!(classify(400, "Prompt too long"), LlmError::ContextOverflow(_)));
        // loose wording with another code is not an overflow
        for body in [r#"{"error":{"code":"1210","message":"context window parameter exceeds the allowed range"}}"#,
                     r#"{"error":{"code":"1211","message":"max length of tokens exceeded"}}"#, "context exceeded"] {
            assert!(matches!(classify(400, body), LlmError::Http { status: 400, .. }), "{body}");
        }
        // an error object inside a 2xx stream follows the same rule
        let loose = r#"{"error":{"code":"1305","message":"context overloaded"}}"#;
        assert!(matches!(stream_error(200, loose), Some(LlmError::Http { status: 200, .. })));
        assert!(matches!(stream_error(200, r#"data: {"error":{"code":1261,"message":"x"}}"#), Some(LlmError::ContextOverflow(_))));
    }
    #[test] fn quota_errors_are_classified_and_not_retried() {
        let LlmError::Quota(msg) = classify(429, QUOTA_BODY_FROM_NOTES) else { panic!("1113 should be Quota") };
        assert_eq!(msg, "Insufficient balance or no resource package. Please recharge.");
        assert_eq!(LlmError::Quota(msg).to_string(), "quota or balance exhausted: Insufficient balance or no resource package. Please recharge.");
        // every 429 code except the transient 1302/1305 is permanent (docs/glm-api-notes.md "Retry caution")
        for code in ["1113", "1308", "1309", "1310", "1311", "1313", "1316", "1321", "9999"] {
            let body = format!(r#"{{"error":{{"code":"{code}","message":"m"}}}}"#);
            assert_eq!(classify(429, &body), LlmError::Quota("m".into()), "{code}");
        }
        assert_eq!(classify(429, r#"{"error":{"code":1316,"message":"m"}}"#), LlmError::Quota("m".into())); // numeric code
        // transient codes, no parseable code, and other statuses stay plain Http, and 429s are still retried
        for body in [r#"{"error":{"code":"1302","message":"Rate limit reached"}}"#, r#"{"error":{"code":1305,"message":"busy"}}"#,
                     r#"{"error":{"message":"no code"}}"#, "<html>Too Many Requests</html>", ""] {
            let e = classify(429, body);
            assert!(matches!(e, LlmError::Http { status: 429, .. }), "{body}");
            assert_eq!(retry_delay(&e, 0, false, None), Some(Duration::from_secs(1)), "{body}");
        }
        assert!(matches!(classify(400, QUOTA_BODY_FROM_NOTES), LlmError::Http { status: 400, .. }));
        assert_eq!(retry_delay(&LlmError::Quota("x".into()), 0, false, None), None);
        assert_eq!(retry_delay(&LlmError::Quota("x".into()), 0, false, Some(5)), None);
    }
    #[test] fn stream_error_detection() {
        let e = r#"{"error":{"code":"1305","message":"busy"}}"#;
        let http200 = Some(LlmError::Http { status: 200, body: e.into() });
        assert_eq!(stream_error(200, &format!("data: {e}\n")), http200);          // in a data: line
        assert_eq!(stream_error(200, &format!("{e}\r\n")), http200);              // bare non-SSE line
        assert!(matches!(stream_error(200, &format!("data: {OVERFLOW_BODY_FROM_NOTES}")), Some(LlmError::ContextOverflow(_))));
        for ok in [r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#, r#"data: {"error":null,"choices":[]}"#,
                   "data: [DONE]", ": keepalive", "", "not json"] {
            assert_eq!(stream_error(200, ok), None, "{ok}");
        }
    }
    #[test] fn fixtures_parse() {
        // Raw bodies captured from the live API (docs/glm-api-notes.md), run through the same line parsing as the client.
        let parse = |raw: &str| finish(run(&sse_chunks(raw)).0);
        let cities = |r: &Reply| r.tool_calls.iter().map(|c| {
            assert_eq!(c.name, "get_time");
            assert!(c.id.starts_with("call_"), "{c:?}");
            serde_json::from_str::<Value>(&c.args).unwrap()["city"].as_str().unwrap().to_string()
        }).collect::<Vec<_>>();

        let text = parse(include_str!("../fixtures/stream_text.sse"));
        assert!(!text.text.is_empty() && !text.reasoning.is_empty() && text.tool_calls.is_empty());
        assert_eq!((text.finish_reason.as_str(), text.usage), ("stop", Usage { prompt: 14, completion: 220, cached: 0 }));

        let tools = parse(include_str!("../fixtures/stream_tools.sse"));
        assert_eq!((tools.finish_reason.as_str(), tools.usage.prompt), ("tool_calls", 177));
        assert_eq!(cities(&tools), ["London"]);

        // whole calls in one chunk each, and index-based partial `arguments` (`tool_stream: true`): same result
        for raw in [include_str!("../fixtures/stream_two_tools.sse"), include_str!("../fixtures/stream_two_tools_partial.sse")] {
            let two = parse(raw);
            assert_eq!(cities(&two), ["Tokyo", "Paris"]);
            assert_ne!(two.tool_calls[0].id, two.tool_calls[1].id);
            assert_eq!((two.finish_reason.as_str(), two.usage.prompt, two.usage.cached), ("tool_calls", 176, 128));
        }
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
        assert_eq!(retry_delay(&LlmError::Quota("x".into()), 0, false, None), None);
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
    #[test] fn effort_cap_only_lowers() {
        assert_eq!(lower_effort(Some("max"), Some("medium")), Some("medium"));
        assert_eq!(lower_effort(Some("low"), Some("medium")), Some("low")); // never raised
        assert_eq!(lower_effort(Some("medium"), Some("medium")), Some("medium"));
        assert_eq!(lower_effort(Some("max"), None), Some("max"));
        assert_eq!(lower_effort(None, None), None); // the API default stays unsent
        assert_eq!(lower_effort(None, Some("low")), Some("low")); // default is max, so low is a cut
        assert_eq!(lower_effort(Some("max"), Some("bogus")), Some("max")); // an unknown cap changes nothing
        assert_eq!(lower_effort(None, Some("bogus")), None);
        assert_eq!(lower_effort(Some("bogus"), Some("low")), Some("bogus")); // nor does an unknown base get replaced
    }
    #[test] fn cap_lands_in_the_request_body() {
        let mut c = ZaiClient::new(ZaiConfig { base_url: "u".into(), model: "m".into(), api_key: "k".into(),
                                               temperature: 1.0, top_p: 0.95, reasoning_effort: Some("max".into()) });
        let effort = |c: &ZaiClient| c.body(&[], &[])["reasoning_effort"].as_str().map(String::from);
        assert_eq!(effort(&c), Some("max".into()));
        c.cap_effort(Some("medium"));
        assert_eq!(effort(&c), Some("medium".into()));
        c.cap_effort(None); // lifting the cap restores the configured effort
        assert_eq!(effort(&c), Some("max".into()));
    }
    #[test] fn stream_needs_done_or_finish_reason() {
        assert!(stream_complete(true, "") && stream_complete(false, "stop"));
        assert!(!stream_complete(false, ""));
    }

    /// Localhost server: accepts one connection, reads the whole request, sends `reply`, holds the socket open for `hold`, then closes.
    fn server(reply: &'static str, hold: Duration) -> ZaiClient { server_parts(vec![(reply, hold)]) }

    /// Like `server`, but writes each part and then sleeps its duration before the next one (the last sleep is the hold).
    fn server_parts(parts: Vec<(&'static str, Duration)>) -> ZaiClient {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", l.local_addr().unwrap());
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let (mut head, mut b) = (Vec::new(), [0u8; 1]);
            while !head.ends_with(b"\r\n\r\n") { s.read_exact(&mut b).unwrap(); head.push(b[0]); }
            let len = String::from_utf8_lossy(&head).to_lowercase().split("content-length:").nth(1)
                .and_then(|v| v.lines().next()?.trim().parse().ok()).unwrap_or(0);
            s.read_exact(&mut vec![0u8; len]).unwrap();
            for (part, pause) in parts {
                s.write_all(part.as_bytes()).unwrap();
                std::thread::sleep(pause);
            }
        });
        ZaiClient::new(ZaiConfig { base_url, model: "m".into(), api_key: "k".into(), temperature: 1.0, top_p: 0.95, reasoning_effort: None })
    }
    const CHUNK_A: &str = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n";

    #[test] fn silent_server_can_be_cancelled() {
        let mut c = server("", Duration::from_secs(3)); // accepts, then never answers
        let cancel = AtomicBool::new(false);
        let t = Instant::now();
        let r = std::thread::scope(|s| {
            s.spawn(|| { std::thread::sleep(Duration::from_millis(200)); cancel.store(true, Ordering::Relaxed); });
            c.complete(&[], &[], &mut |_| {}, &cancel)
        }).unwrap();
        assert!(r.interrupted && t.elapsed() < Duration::from_secs(2));
    }
    #[test] fn cancel_keeps_the_partial_reasoning() {
        // What a SIGINT from `timeout` relies on: the thinking streamed so far still reaches the trace.
        let mut c = server("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"half a thought\"}}]}\n\n",
                           Duration::from_secs(3));
        let cancel = AtomicBool::new(false);
        let r = c.complete(&[], &[], &mut |d| if let Delta::Reasoning(_) = d { cancel.store(true, Ordering::Relaxed) }, &cancel).unwrap();
        assert_eq!((r.interrupted, r.reasoning.as_str()), (true, "half a thought"));
    }
    // A reply is the HTTP head plus its first chunk; ANSWER can follow as a later part.
    const THINKING: &str = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"half a thought\"}}]}\n\n";
    const ANSWERED: &str = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}]}\n\n";
    const ANSWER: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}]}\n\n";

    #[test] fn think_ms_runs_from_the_first_reasoning_to_the_first_output() {
        let mut c = server_parts(vec![(THINKING, Duration::from_millis(400)), (ANSWER, Duration::ZERO)]);
        let r = c.complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!(r.text, "b");
        assert!((400..2000).contains(&r.think_ms.unwrap()), "{:?}", r.think_ms);
        // no reasoning at all: no thinking time
        let r = server(ANSWERED, Duration::ZERO).complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!(r.think_ms, Some(0));
    }
    #[test] fn think_ms_is_kept_when_the_call_is_cancelled() {
        let mut c = server(THINKING, Duration::from_secs(3));
        let cancel = AtomicBool::new(false);
        let r = std::thread::scope(|s| {
            s.spawn(|| { std::thread::sleep(Duration::from_millis(500)); cancel.store(true, Ordering::Relaxed); });
            c.complete(&[], &[], &mut |_| {}, &cancel)
        }).unwrap();
        assert!(r.interrupted && r.think_ms.is_some_and(|ms| ms >= 300), "{:?}", r.think_ms);
    }
    #[test] fn reply_carries_the_effort_sent() {
        let run = |base: Option<&str>, cap: Option<&str>| {
            let mut c = server(ANSWERED, Duration::ZERO);
            c.cfg.reasoning_effort = base.map(String::from);
            c.cap_effort(cap);
            c.complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap().effort
        };
        assert_eq!(run(Some("max"), None).as_deref(), Some("max"));
        assert_eq!(run(Some("max"), Some("medium")).as_deref(), Some("medium"));
        assert_eq!(run(Some("low"), Some("medium")).as_deref(), Some("low")); // a cap never raises
        assert_eq!(run(None, None), None); // nothing sent: the API default applied
    }
    const TEXT_CHUNK: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n";
    const CALL_CHUNK: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}\n\n";
    const NO_DONE: &str = "stream ended before [DONE]"; // what a stream that is simply left to run out reports

    #[test] fn thinking_cap_cuts_a_silent_thought_and_keeps_it() {
        let mut c = server(THINKING, Duration::from_secs(3));
        c.think_cap(Some(Duration::from_millis(400)));
        let t = Instant::now();
        let r = c.complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap();
        let waited = t.elapsed();
        assert!(r.capped && !r.interrupted);
        assert_eq!((r.reasoning.as_str(), r.text.as_str(), r.tool_calls.len()), ("half a thought", "", 0));
        assert!(waited >= Duration::from_millis(400) && waited < Duration::from_secs(2), "{waited:?}"); // not at once, not at the server's pace
        assert!(r.think_ms.is_some_and(|ms| (400..2000).contains(&ms)), "{:?}", r.think_ms);
    }
    #[test] fn no_cap_means_no_cut() {
        // the same stream with no cap runs until the server ends it
        let e = server(THINKING, Duration::from_millis(700)).complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
        assert_eq!(e, LlmError::Network(NO_DONE.into()));
    }
    #[test] fn thinking_cap_never_fires_once_content_or_a_tool_call_has_started() {
        for first_output in [TEXT_CHUNK, CALL_CHUNK] {
            let mut c = server_parts(vec![(THINKING, Duration::ZERO), (first_output, Duration::from_millis(900))]);
            c.think_cap(Some(Duration::from_millis(300))); // long gone by the time the server ends the stream
            let e = c.complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
            assert_eq!(e, LlmError::Network(NO_DONE.into()), "{first_output}");
        }
    }
    #[test] fn thinking_cap_waits_for_reasoning_and_leaves_stalls_to_the_idle_limit() {
        // a silent server is no thought: the cap stays out of it and only the cancel ends the call
        let mut c = server("", Duration::from_secs(3));
        c.think_cap(Some(Duration::from_millis(100)));
        let cancel = AtomicBool::new(false);
        let r = std::thread::scope(|s| {
            s.spawn(|| { std::thread::sleep(Duration::from_millis(600)); cancel.store(true, Ordering::Relaxed); });
            c.complete(&[], &[], &mut |_| {}, &cancel)
        }).unwrap();
        assert!(r.interrupted && !r.capped && r.reasoning.is_empty());
        // reasoning, then silence longer than the idle limit but shorter than the cap: the idle limit wins as before
        let mut c = server(THINKING, Duration::from_secs(3));
        c.idle = Duration::from_millis(300);
        c.think_cap(Some(Duration::from_secs(60)));
        let e = c.complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
        assert_eq!(e, LlmError::Network("stream stalled".into()));
    }
    #[test] fn cancel_wins_over_the_thinking_cap() {
        // Ctrl+C and the cap due at the same moment: the call is interrupted, not capped
        let mut c = server(THINKING, Duration::from_secs(3));
        c.think_cap(Some(Duration::ZERO));
        let cancel = AtomicBool::new(false);
        let r = c.complete(&[], &[], &mut |d| if let Delta::Reasoning(_) = d { cancel.store(true, Ordering::Relaxed) }, &cancel).unwrap();
        assert_eq!((r.interrupted, r.capped, r.reasoning.as_str()), (true, false, "half a thought"));
    }
    #[test] fn silent_server_trips_idle_timeout_and_retries() {
        let mut c = server("", Duration::from_secs(3));
        c.idle = Duration::from_millis(300);
        let (cancel, mut notices) = (AtomicBool::new(false), Vec::new());
        let r = c.complete(&[], &[], &mut |d| if let Delta::Notice(n) = d { notices.push(n); cancel.store(true, Ordering::Relaxed) }, &cancel).unwrap();
        assert_eq!(notices, ["network error: stream stalled; retrying in 1s"]); // nothing emitted, so retryable
        assert!(r.interrupted);
    }
    #[test] fn eof_without_done_or_finish_reason_is_an_error() {
        // text already streamed: not retryable, surfaces as Network
        let e = server(CHUNK_A, Duration::ZERO).complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
        assert_eq!(e, LlmError::Network("stream ended before [DONE]".into()));
        // headers only, nothing emitted: retried
        let (cancel, mut notices) = (AtomicBool::new(false), Vec::new());
        server("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", Duration::ZERO)
            .complete(&[], &[], &mut |d| if let Delta::Notice(n) = d { notices.push(n); cancel.store(true, Ordering::Relaxed) }, &cancel).unwrap();
        assert_eq!(notices, ["network error: stream ended before [DONE]; retrying in 1s"]);
    }
    #[test] fn http_errors_and_done_flow_through_worker() {
        let run = |reply: &'static str| server(reply, Duration::ZERO).complete(&[], &[], &mut |_| {}, &AtomicBool::new(false));
        assert!(matches!(run("HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\nConnection: close\r\n\r\nno"), Err(LlmError::Auth(_))));
        assert!(matches!(run("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: {\"error\":{\"code\":\"1305\"}}\n\n"),
                         Err(LlmError::Http { status: 200, .. })));
        let r = run("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n: ka\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: [DONE]\n\n").unwrap();
        assert_eq!((r.text.as_str(), r.interrupted), ("a", false));
    }
    #[test] fn eof_after_finish_reason_is_ok() {
        let reply = concat!("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n");
        let r = server(reply, Duration::ZERO).complete(&[], &[], &mut |_| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!((r.text.as_str(), r.finish_reason.as_str(), r.interrupted), ("a", "stop", false));
    }
    #[test] fn backoff_sleep_honors_cancel() {
        assert!(wait(Duration::ZERO, &AtomicBool::new(false)));
        let t = std::time::Instant::now();
        assert!(!wait(Duration::from_secs(30), &AtomicBool::new(true)));
        assert!(t.elapsed() < Duration::from_secs(1));
    }
}

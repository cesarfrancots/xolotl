//! The turn loop (spec §11, §12): for each user request, build the view from the events, stream the model call,
//! run tool calls through permissions, record every step as an event, and apply the loop guards.

use crate::llm::{Delta, LlmError, Model, Reply, ToolCall};
use crate::perms::{Perms, Verdict};
use crate::policy::{self, LIMITS};
use crate::store::{Kind, Session};
use crate::tools::{self, ToolCtx};
use crate::ui::{Decision, PermissionRequest, Status, Ui, mini_diff, tool_summary};
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const CANCELLED: &str = "cancelled by user";
const HEADLESS_DENIED: &str = "denied: not allowed in headless mode (use --mode or --allow)";
const EMPTY_NUDGE: &str = "(empty response — continue the task, or reply with your answer)";
/// Lines of an edit/write diff shown in a permission prompt.
const DIFF_LINES: usize = 20;

pub struct Agent<'a> {
    pub session: Session,
    pub model: &'a mut dyn Model,
    pub ui: &'a mut dyn Ui,
    pub perms: Perms,
    pub tools: ToolCtx,
    pub budget: usize,
    pub cancel: &'a AtomicBool,
}

#[derive(Debug, PartialEq)]
pub enum Outcome {
    Done,
    StepLimit,
    Interrupted,
    Error(String),
}

/// One finished call as the repeat guard compares it.
#[derive(PartialEq)]
struct Seen {
    name: String,
    args: Value,
    ok: bool,
    exit_code: Option<i32>,
    body: String,
}

impl Agent<'_> {
    /// Runs one user request to its end, then shows the status line (on every return path).
    pub fn run_request(&mut self, text: &str) -> Outcome {
        let start = Instant::now();
        let mut tool_calls = 0;
        let outcome = self.turn(text, &mut tool_calls).unwrap_or_else(|failed| failed);
        let usage = self.session.events.iter().rev().find_map(|e| match &e.kind {
            Kind::Assistant { usage, .. } => Some(*usage),
            _ => None,
        });
        let cached_pct = usage.filter(|u| u.prompt > 0).map_or(0, |u| (u.cached * 100 / u.prompt).min(100) as u8);
        self.ui.status(&Status {
            ctx_tokens: policy::estimate_tokens(&self.session.events),
            budget: self.budget,
            tool_calls,
            elapsed_ms: start.elapsed().as_millis() as u64,
            cached_pct,
        });
        outcome
    }

    /// Stage-2 compaction (spec §10.4).
    // ponytail: stub until Task 12 wires compaction in; until then a context overflow ends the request.
    pub fn compact(&mut self, _focus: Option<&str>) -> Result<(), String> {
        Err("compaction not implemented".into())
    }

    /// The loop of spec §11. `Err` is a failed session write, which ends the request at once: nothing may be
    /// appended after a partial write.
    fn turn(&mut self, request: &str, tool_calls: &mut usize) -> Result<Outcome, Outcome> {
        self.cancel.store(false, Ordering::SeqCst);
        self.append(Kind::User { text: request.into() })?;
        let schemas = tools::schemas();
        let (mut steps, mut check_fired, mut edited, mut empty_retried, mut overflow_retried) = (0, false, false, false, false);
        let mut recent: Vec<Seen> = Vec::new(); // tool results since the last repeat nudge
        loop {
            // Ctrl+C during the last tool of a batch (or at its prompt) must not send another model request
            if self.cancel.load(Ordering::SeqCst) {
                return self.interrupt(&[], "tool", None);
            }
            if steps == LIMITS.step_limit {
                if !self.ui.confirm(&format!("step limit reached ({}) — continue?", LIMITS.step_limit)) {
                    return Ok(Outcome::StepLimit);
                }
                steps = 0;
            }
            let view = policy::build_view(&self.session.events);
            let t0 = Instant::now();
            let ui = &mut *self.ui;
            let result = self.model.complete(
                &view,
                &schemas,
                &mut |d| match d {
                    Delta::Text(t) => ui.text_delta(t),
                    Delta::Reasoning(r) => ui.reasoning_delta(r),
                    Delta::Notice(n) => ui.notice(&n),
                },
                self.cancel,
            );
            let reply = match result {
                Ok(reply) => reply,
                Err(LlmError::ContextOverflow(_)) if !overflow_retried => {
                    overflow_retried = true;
                    if let Err(e) = self.compact(None) {
                        self.append(Kind::Error { kind: "compaction".into(), message: e.clone(), retry: false })?;
                        self.ui.notice(&e);
                        return Ok(Outcome::Error(e));
                    }
                    continue;
                }
                Err(e) => {
                    let msg = match e {
                        LlmError::Auth(_) => "authentication failed — check ZAI_API_KEY".to_string(),
                        _ => e.to_string(),
                    };
                    self.append(Kind::Error { kind: "llm".into(), message: e.to_string(), retry: false })?;
                    self.ui.notice(&msg);
                    return Ok(Outcome::Error(msg));
                }
            };
            steps += 1;
            *tool_calls += reply.tool_calls.len();
            let latency_ms = t0.elapsed().as_millis() as u64;
            let Reply { text, reasoning, tool_calls: calls, usage, finish_reason, interrupted } = reply;
            if interrupted {
                let text = format!("{text}\n[interrupted]");
                self.append(Kind::Assistant { text, reasoning, tool_calls: calls.clone(), usage, finish_reason, latency_ms })?;
                return self.interrupt(&calls, "stream", None);
            }
            self.append(Kind::Assistant { text: text.clone(), reasoning, tool_calls: calls.clone(), usage, finish_reason, latency_ms })?;

            if calls.is_empty() && text.trim().is_empty() {
                if empty_retried {
                    self.ui.notice("empty response");
                    return Ok(Outcome::Error("empty response".into()));
                }
                empty_retried = true;
                self.append(nudge("empty", EMPTY_NUDGE.into()))?;
                continue;
            }
            empty_retried = false; // only back-to-back empty replies end the request (spec §11.4)
            if calls.is_empty() {
                if edited && !check_fired {
                    check_fired = true;
                    self.append(nudge("completion_check", policy::completion_check(request)))?;
                    continue;
                }
                return Ok(Outcome::Done);
            }

            for (i, call) in calls.iter().enumerate() {
                if self.cancel.load(Ordering::SeqCst) {
                    return self.interrupt(&calls[i..], "tool", Some(&call.id));
                }
                let (seen, cancelled) = self.handle(call)?;
                edited |= seen.ok && matches!(seen.name.as_str(), "edit" | "write");
                recent.push(seen);
                if cancelled {
                    return self.interrupt(&calls[i + 1..], "tool", Some(&call.id));
                }
            }

            let n = LIMITS.repeat_threshold;
            if recent.len() >= n && recent[recent.len() - n..].windows(2).all(|w| w[0] == w[1]) {
                self.append(nudge("repeat", format!("this call returned the same result {n} times — change approach")))?;
                recent.clear();
            }
            recent.drain(..recent.len().saturating_sub(n));
        }
    }

    /// Validates, gates and runs one call, appending its `permission` and `tool_result` events. Returns what the
    /// repeat guard compares, and whether the tool was cancelled while running. Only a call with a known tool
    /// name and valid arguments reaches the Ui.
    fn handle(&mut self, call: &ToolCall) -> Result<(Seen, bool), Outcome> {
        let args = match serde_json::from_str::<Value>(&call.args) {
            Ok(args) => args,
            Err(e) => return self.refuse(call, Value::String(call.args.clone()), format!("invalid JSON arguments: {e}")),
        };
        if !tools::TOOL_NAMES.contains(&call.name.as_str()) {
            let msg = format!("unknown tool `{}`; available: {}", call.name, tools::TOOL_NAMES.join(", "));
            return self.refuse(call, args, msg);
        }
        let decision = match self.perms.check(&call.name, &args) {
            Verdict::Allow("mode") => None, // the mode's default for this tool: not a decision worth an event
            Verdict::Allow(by) => Some(("allow", by)),
            Verdict::Deny(by) => Some(("deny", by)),
            Verdict::Ask => Some(match self.ui.ask(&permission_request(call, &args)) {
                Decision::Allow => ("allow", "user"),
                Decision::Deny => ("deny", "user"),
                Decision::Always => {
                    self.perms.grant(&call.name, &args);
                    ("always", "user")
                }
            }),
        };
        if let Some((decision, by)) = decision {
            self.append(Kind::Permission { call_id: call.id.clone(), decision: decision.into(), by: by.into() })?;
            if decision == "deny" {
                let msg = if by == "user" { "denied by user" } else { HEADLESS_DENIED };
                return self.refuse(call, args, msg.into());
            }
        }

        self.ui.tool_start(call);
        let t0 = Instant::now();
        let out = tools::run(&mut self.tools, &call.name, &call.id, &args, self.cancel);
        let duration_ms = t0.elapsed().as_millis() as u64;
        let capped = policy::cap_output(&call.name, &out.text);
        let (output, artifact) = if !capped.truncated {
            (capped.text, None)
        } else {
            match self.session.write_artifact(&format!("{}.txt", call.id), out.text.as_bytes()) {
                Ok(path) => (capped.text + &truncation_note(out.text.len(), &path), Some(path.display().to_string())),
                Err(e) => (format!("{}\n[truncated: the full output could not be saved: {e}]", capped.text), None),
            }
        };
        self.append(Kind::ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            ok: out.ok,
            output,
            artifact,
            bytes: out.text.len() as u64,
            duration_ms,
            exit_code: out.exit_code,
        })?;
        self.ui.tool_end(call, &out, duration_ms);
        // `bash` ends with a timing footer (`[exit 0 · 0.3s]`) that differs between otherwise identical runs
        let body = if call.name == "bash" { out.text.rsplit_once('\n').map_or("", |(body, _)| body) } else { &out.text };
        let seen = Seen { name: call.name.clone(), args, ok: out.ok, exit_code: out.exit_code, body: body.to_string() };
        Ok((seen, out.cancelled))
    }

    /// Records a call that is not run: its result is `msg`.
    fn refuse(&mut self, call: &ToolCall, args: Value, msg: String) -> Result<(Seen, bool), Outcome> {
        self.append(failed(call, &msg))?;
        Ok((Seen { name: call.name.clone(), args, ok: false, exit_code: None, body: msg }, false))
    }

    /// Gives each of `calls` a `cancelled by user` result (so every call stays paired), then records the interrupt.
    fn interrupt(&mut self, calls: &[ToolCall], during: &str, call_id: Option<&str>) -> Result<Outcome, Outcome> {
        for call in calls {
            self.append(failed(call, CANCELLED))?;
        }
        self.append(Kind::Interrupt { during: during.into(), call_id: call_id.map(Into::into) })?;
        Ok(Outcome::Interrupted)
    }

    /// Every event goes to disk, then to the Ui.
    fn append(&mut self, kind: Kind) -> Result<(), Outcome> {
        let e = self.session.append(kind).map_err(|e| Outcome::Error(format!("cannot write session: {e}")))?;
        self.ui.event(&e);
        Ok(())
    }
}

/// `"\n[truncated: full output {KB}KB at {path} — use read/grep]"`, KB rounded up.
pub(crate) fn truncation_note(bytes: usize, path: &Path) -> String {
    format!("\n[truncated: full output {}KB at {} — use read/grep]", bytes.div_ceil(1024), path.display())
}

fn nudge(reason: &str, text: String) -> Kind {
    Kind::Nudge { reason: reason.into(), text }
}

fn failed(call: &ToolCall, msg: &str) -> Kind {
    Kind::ToolResult {
        call_id: call.id.clone(),
        name: call.name.clone(),
        ok: false,
        output: msg.into(),
        artifact: None,
        bytes: msg.len() as u64,
        duration_ms: 0,
        exit_code: None,
    }
}

/// The summary is one clipped line, so `bash` shows the whole command as the detail (the user must see exactly
/// what they approve), and `edit`/`write` show a diff.
fn permission_request(call: &ToolCall, args: &Value) -> PermissionRequest {
    let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("");
    let detail = match call.name.as_str() {
        "bash" => Some(s("command").to_string()),
        "edit" => Some(mini_diff(s("old"), s("new"), DIFF_LINES)),
        "write" => Some(mini_diff("", s("content"), DIFF_LINES)),
        _ => None,
    };
    PermissionRequest { tool: call.name.clone(), summary: tool_summary(call), detail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, Message, Reply, Scripted, ToolCall, Usage};
    use crate::perms::Mode;
    use crate::store::{Event, Meta, now_ms, test_dir};
    use crate::tools::{self, Shell, ToolCtx, ToolOut};
    use crate::ui::{Decision, PermissionRequest, Status, Ui};
    use serde_json::json;
    use std::collections::{HashMap, VecDeque};
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct TestUi<'c> {
        events: Vec<Event>,
        notices: Vec<String>,
        asks: Vec<PermissionRequest>,
        questions: Vec<String>,
        statuses: Vec<Status>,
        text: String,
        started: Vec<String>,
        ended: Vec<String>,
        decisions: VecDeque<Decision>,
        confirms: VecDeque<bool>,
        cancel_on_tool_start: Option<&'c AtomicBool>,
        cancel_on_tool_end: Option<&'c AtomicBool>,
        /// Like Ctrl+C at a Terminal prompt: sets the flag and answers Deny.
        cancel_on_ask: Option<&'c AtomicBool>,
    }

    impl Ui for TestUi<'_> {
        fn text_delta(&mut self, s: &str) {
            self.text.push_str(s);
        }
        fn reasoning_delta(&mut self, _: &str) {}
        fn tool_start(&mut self, call: &ToolCall) {
            self.started.push(call.id.clone());
            if let Some(c) = self.cancel_on_tool_start {
                c.store(true, Ordering::SeqCst);
            }
        }
        fn tool_end(&mut self, call: &ToolCall, _: &ToolOut, _: u64) {
            self.ended.push(call.id.clone());
            if let Some(c) = self.cancel_on_tool_end {
                c.store(true, Ordering::SeqCst);
            }
        }
        fn ask(&mut self, req: &PermissionRequest) -> Decision {
            self.asks.push(req.clone());
            if let Some(c) = self.cancel_on_ask {
                c.store(true, Ordering::SeqCst);
                return Decision::Deny;
            }
            self.decisions.pop_front().unwrap_or(Decision::Deny)
        }
        fn confirm(&mut self, question: &str) -> bool {
            self.questions.push(question.into());
            self.confirms.pop_front().unwrap_or(false)
        }
        fn event(&mut self, e: &Event) {
            self.events.push(e.clone());
        }
        fn notice(&mut self, msg: &str) {
            self.notices.push(msg.into());
        }
        fn status(&mut self, s: &Status) {
            self.statuses.push(s.clone());
        }
    }

    fn text(s: &str) -> Result<Reply, LlmError> {
        Ok(Reply { text: s.into(), finish_reason: "stop".into(), ..Default::default() })
    }
    fn tc(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall { id: id.into(), name: name.into(), args: args.into() }
    }
    fn calls(cs: Vec<ToolCall>) -> Result<Reply, LlmError> {
        Ok(Reply { tool_calls: cs, finish_reason: "tool_calls".into(), ..Default::default() })
    }
    fn call(id: &str, name: &str, args: &str) -> Result<Reply, LlmError> {
        calls(vec![tc(id, name, args)])
    }
    fn read_a(id: &str) -> Result<Reply, LlmError> {
        call(id, "read", r#"{"path":"a.txt"}"#)
    }

    /// An agent over a fresh session (`SessionStart{system:"S"}`), rooted at an empty temp project dir.
    fn agent<'a>(tag: &str, mode: Mode, headless: bool, model: &'a mut Scripted, ui: &'a mut TestUi, cancel: &'a AtomicBool) -> Agent<'a> {
        let root = test_dir(tag);
        let proj = root.join("proj");
        fs::create_dir_all(&proj).unwrap();
        let cwd = proj.display().to_string();
        let meta = Meta {
            id: "20260101-000000-0001".into(),
            cwd: cwd.clone(),
            git_head: None,
            model: "m".into(),
            mode: mode.as_str().into(),
            policy_hash: "h".into(),
            started_ms: now_ms(),
            ended_ms: None,
            status: "running".into(),
        };
        let mut session = Session::create(&root.join("home"), meta).unwrap();
        session
            .append(Kind::SessionStart { cwd, model: "m".into(), mode: mode.as_str().into(), policy_hash: "h".into(), system: "S".into() })
            .unwrap();
        let tools = ToolCtx { root: proj, shell: tools::resolve_shell(), backups_dir: session.artifacts_dir(), reads: HashMap::new() };
        let perms = Perms::new(mode, headless, session.dir.clone(), vec![]);
        Agent { session, model, ui, perms, tools, budget: 200_000, cancel }
    }

    fn has_bash() -> bool {
        matches!(tools::resolve_shell(), Shell::Bash(_))
    }
    fn nudges(events: &[Event], reason: &str) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::Nudge { reason: r, text } if r == reason => Some(text.clone()),
                _ => None,
            })
            .collect()
    }
    /// `(call_id, ok, output)` of every tool result.
    fn results(events: &[Event]) -> Vec<(String, bool, String)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::ToolResult { call_id, ok, output, .. } => Some((call_id.clone(), *ok, output.clone())),
                _ => None,
            })
            .collect()
    }
    /// `(call_id, decision, by)` of every permission event.
    fn permissions(events: &[Event]) -> Vec<(String, String, String)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::Permission { call_id, decision, by } => Some((call_id.clone(), decision.clone(), by.clone())),
                _ => None,
            })
            .collect()
    }
    fn interrupts(events: &[Event]) -> Vec<(String, Option<String>)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::Interrupt { during, call_id } => Some((during.clone(), call_id.clone())),
                _ => None,
            })
            .collect()
    }
    fn last_user(req: &(Vec<Message>, usize)) -> String {
        match req.0.last() {
            Some(Message::User(t)) => t.clone(),
            other => panic!("last message is not a user message: {other:?}"),
        }
    }

    #[test]
    fn read_edit_check_done() {
        let mut m = Scripted::new(vec![read_a("c1"), call("c2", "edit", r#"{"path":"a.txt","old":"hello","new":"bye"}"#), text("done"), text("verified")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(true)); // a stale Ctrl+C must not cancel the new request
        let mut a = agent("rec", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("change hello to bye"), Outcome::Done);
        let Agent { session, tools, .. } = a;

        assert_eq!(fs::read_to_string(tools.root.join("a.txt")).unwrap(), "bye\n");
        assert_eq!(nudges(&session.events, "completion_check").len(), 1);
        assert_eq!(m.requests.len(), 4);
        let last = last_user(&m.requests[3]);
        assert!(last.contains("<request>") && last.contains("change hello to bye"), "{last}");
        assert_eq!(m.requests[0].1, tools::schemas().len()); // every call offers the tools
        assert_eq!(ui.statuses.len(), 1);
        assert_eq!((ui.statuses[0].tool_calls, ui.statuses[0].budget), (2, 200_000));
        assert_eq!(ui.started, ["c1", "c2"]);
    }

    #[test]
    fn no_check_without_edits() {
        let reply = Reply { usage: Usage { prompt: 200, completion: 5, cached: 50 }, ..text("hi").unwrap() };
        let mut m = Scripted::new(vec![Ok(reply)]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("nocheck", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("say hi"), Outcome::Done);
        let session = a.session;

        assert!(session.events.iter().all(|e| !matches!(e.kind, Kind::Nudge { .. })));
        assert_eq!(m.requests.len(), 1);
        assert_eq!(ui.text, "hi"); // text deltas reach the Ui
        let s = &ui.statuses[0];
        assert_eq!((s.tool_calls, s.cached_pct, s.ctx_tokens), (0, 25, policy::estimate_tokens(&session.events)));
    }

    #[test]
    fn completion_check_fires_once() {
        if !has_bash() {
            return;
        }
        let mut m = Scripted::new(vec![
            read_a("c1"),
            call("c2", "edit", r#"{"path":"a.txt","old":"hello","new":"bye"}"#),
            text("done"),
            call("c3", "bash", r#"{"command":"echo ok"}"#),
            text("all good"),
        ]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("checkonce", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("change it"), Outcome::Done);
        let session = a.session;

        assert_eq!(nudges(&session.events, "completion_check").len(), 1);
        assert_eq!(m.requests.len(), 5);
        let bash = &results(&session.events)[2];
        assert!(bash.1 && bash.2.starts_with("ok\n"), "{bash:?}");
    }

    #[test]
    fn invalid_json_args() {
        let mut m = Scripted::new(vec![call("c1", "read", "{bad"), text("sorry")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("badjson", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("read"), Outcome::Done);
        let session = a.session;

        let r = results(&session.events);
        assert_eq!(r.len(), 1);
        assert!(!r[0].1 && r[0].2.starts_with("invalid JSON arguments"), "{r:?}");
        assert!(ui.started.is_empty() && ui.ended.is_empty());
        // the model sees the error as the call's result
        assert!(matches!(&m.requests[1].0.last(), Some(Message::Tool { call_id, content }) if call_id == "c1" && content.starts_with("invalid JSON")));
    }

    #[test]
    fn unknown_tool() {
        let mut m = Scripted::new(vec![call("c1", "fly", "{}"), text("ok")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("unknown", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("fly"), Outcome::Done);
        let session = a.session;

        let r = results(&session.events);
        assert_eq!(r.len(), 1);
        assert!(!r[0].1 && r[0].2.starts_with("unknown tool `fly`; available: read, write"), "{r:?}");
        // a raw model-chosen name never reaches the Ui
        assert!(ui.started.is_empty() && ui.ended.is_empty() && ui.asks.is_empty());
        assert!(permissions(&session.events).is_empty());
    }

    #[test]
    fn headless_denial_continues() {
        let mut m = Scripted::new(vec![call("c1", "bash", r#"{"command":"ls"}"#), text("ok")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("headless", Mode::Writes, true, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("list"), Outcome::Done);
        let session = a.session;

        assert_eq!(permissions(&session.events), [("c1".to_string(), "deny".to_string(), "mode".to_string())]);
        assert_eq!(results(&session.events), [("c1".to_string(), false, "denied: not allowed in headless mode (use --mode or --allow)".to_string())]);
        assert!(ui.asks.is_empty() && ui.started.is_empty());
        assert_eq!(m.requests.len(), 2);
    }

    #[test]
    fn ask_always_grants() {
        let mut m = Scripted::new(vec![
            read_a("c1"),
            call("c2", "edit", r#"{"path":"a.txt","old":"hello","new":"bye"}"#),
            call("c3", "edit", r#"{"path":"a.txt","old":"bye","new":"ciao"}"#),
            text("done"),
            text("checked"),
        ]);
        let (mut ui, cancel) = (TestUi { decisions: [Decision::Always].into(), ..Default::default() }, AtomicBool::new(false));
        let mut a = agent("always", Mode::Writes, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("edit twice"), Outcome::Done);
        let Agent { session, tools, .. } = a;

        assert_eq!(fs::read_to_string(tools.root.join("a.txt")).unwrap(), "ciao\n");
        assert_eq!(ui.asks.len(), 1);
        let req = &ui.asks[0];
        assert_eq!((req.tool.as_str(), req.summary.as_str(), req.detail.as_deref()), ("edit", "a.txt", Some("- hello\n+ bye")));
        let p = |id: &str, d: &str, by: &str| (id.to_string(), d.to_string(), by.to_string());
        assert_eq!(permissions(&session.events), [p("c2", "always", "user"), p("c3", "allow", "rule")]);
    }

    #[test]
    fn user_denial() {
        let bash = tc("c1", "bash", &json!({ "command": "echo hi\nrm -rf x" }).to_string());
        let write = tc("c2", "write", r#"{"path":"b.txt","content":"x\ny"}"#);
        let mut m = Scripted::new(vec![calls(vec![bash, write]), text("ok")]);
        let (mut ui, cancel) = (TestUi { decisions: [Decision::Deny, Decision::Deny].into(), ..Default::default() }, AtomicBool::new(false));
        let mut a = agent("deny", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("go"), Outcome::Done);
        let Agent { session, tools, .. } = a;

        let r = results(&session.events);
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|(_, ok, out)| !ok && out == "denied by user"), "{r:?}");
        assert!(!tools.root.join("b.txt").exists() && ui.started.is_empty());
        assert_eq!(permissions(&session.events).iter().map(|p| (p.1.as_str(), p.2.as_str())).collect::<Vec<_>>(), [("deny", "user"); 2]);
        assert_eq!(m.requests.len(), 2); // the loop went on
        // bash: the summary is the first line only, so the detail carries the whole command; write: a diff
        assert_eq!((ui.asks[0].summary.as_str(), ui.asks[0].detail.as_deref()), ("echo hi", Some("echo hi\nrm -rf x")));
        assert_eq!((ui.asks[1].summary.as_str(), ui.asks[1].detail.as_deref()), ("b.txt", Some("+ x\n+ y")));
    }

    #[test]
    fn interrupt_mid_stream() {
        let reply = Reply { text: "partial".into(), tool_calls: vec![tc("c1", "read", r#"{"path":"a.txt"}"#)], interrupted: true, ..Default::default() };
        let mut m = Scripted::new(vec![Ok(reply), text("never")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("stream", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("go"), Outcome::Interrupted);
        let session = a.session;

        let assistant = session.events.iter().find_map(|e| match &e.kind {
            Kind::Assistant { text, tool_calls, .. } => Some((text.clone(), tool_calls.len())),
            _ => None,
        });
        assert_eq!(assistant, Some(("partial\n[interrupted]".to_string(), 1)));
        assert_eq!(results(&session.events), [("c1".to_string(), false, "cancelled by user".to_string())]);
        assert_eq!(interrupts(&session.events), [("stream".to_string(), None)]);
        assert!(ui.started.is_empty());
        assert_eq!(m.requests.len(), 1);
        // the view pairs the call with its recorded result, not a synthetic one
        let view = policy::build_view(&session.events);
        let i = view.iter().position(|m| matches!(m, Message::Assistant { .. })).unwrap();
        assert_eq!(view.get(i + 1), Some(&Message::Tool { call_id: "c1".into(), content: "cancelled by user".into() }));
    }

    #[test]
    fn cancel_between_tools() {
        let cancel = AtomicBool::new(false);
        let mut m = Scripted::new(vec![calls(vec![tc("c1", "read", r#"{"path":"a.txt"}"#), tc("c2", "read", r#"{"path":"a.txt"}"#)]), text("never")]);
        let mut ui = TestUi { cancel_on_tool_end: Some(&cancel), ..Default::default() };
        let mut a = agent("between", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("go"), Outcome::Interrupted);
        let session = a.session;

        let r = results(&session.events);
        assert_eq!(r.len(), 2);
        assert!(r[0].1);
        assert_eq!(r[1], ("c2".to_string(), false, "cancelled by user".to_string()));
        assert_eq!(interrupts(&session.events), [("tool".to_string(), Some("c2".to_string()))]);
        assert_eq!((ui.started.as_slice(), m.requests.len()), (&["c1".to_string()][..], 1));
    }

    #[test]
    fn cancel_during_tool_cancels_the_rest() {
        if !has_bash() {
            return;
        }
        let cancel = AtomicBool::new(false);
        let mut m = Scripted::new(vec![calls(vec![tc("c1", "bash", r#"{"command":"sleep 5"}"#), tc("c2", "read", r#"{"path":"a.txt"}"#)]), text("never")]);
        let mut ui = TestUi { cancel_on_tool_start: Some(&cancel), ..Default::default() };
        let mut a = agent("during", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        let t = Instant::now();
        assert_eq!(a.run_request("go"), Outcome::Interrupted);
        assert!(t.elapsed() < Duration::from_secs(4));
        let session = a.session;

        let r = results(&session.events);
        assert!(!r[0].1 && r[0].2.ends_with("[cancelled by user]"), "{r:?}");
        assert_eq!(r[1], ("c2".to_string(), false, "cancelled by user".to_string()));
        assert_eq!(interrupts(&session.events), [("tool".to_string(), Some("c1".to_string()))]);
        assert_eq!(ui.ended, ["c1"]);
    }

    #[test]
    fn cancel_after_last_call_sends_no_request() {
        // Ctrl+C at the prompt of a reply's only call (denied), and during its only, non-cancellable tool (read runs)
        for at_prompt in [true, false] {
            let cancel = AtomicBool::new(false);
            let (tag, script, mut ui) = if at_prompt {
                ("cancelask", call("c1", "bash", r#"{"command":"ls"}"#), TestUi { cancel_on_ask: Some(&cancel), ..Default::default() })
            } else {
                ("cancelend", read_a("c1"), TestUi { cancel_on_tool_end: Some(&cancel), ..Default::default() })
            };
            let mut m = Scripted::new(vec![script, text("never")]);
            let mut a = agent(tag, Mode::Writes, false, &mut m, &mut ui, &cancel);
            fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
            assert_eq!(a.run_request("go"), Outcome::Interrupted, "{tag}");
            let session = a.session;

            assert_eq!(m.requests.len(), 1, "{tag}: no model request after the cancel");
            let r = results(&session.events);
            assert_eq!((r.len(), r[0].1), (1, !at_prompt), "{tag}: {r:?}");
            let fake = session.events.iter().any(|e| matches!(&e.kind, Kind::Assistant { text, .. } if text.contains("[interrupted]")));
            assert!(!fake, "{tag}: no fake interrupted assistant turn");
            assert_eq!(interrupts(&session.events), [("tool".to_string(), None)], "{tag}");
            assert_eq!(ui.statuses.len(), 1, "{tag}");
        }
    }

    #[test]
    fn empty_responses_apart_each_get_a_nudge() {
        let mut m = Scripted::new(vec![text(""), read_a("c1"), text(""), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("emptyapart", Mode::Writes, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("go"), Outcome::Done);
        let session = a.session;

        assert_eq!(nudges(&session.events, "empty").len(), 2);
        assert_eq!(m.requests.len(), 4);
    }

    #[test]
    fn repeat_nudge() {
        // five identical calls: the guard fires at the third and re-arms, so the fourth and fifth do not fire it again
        let mut script: Vec<_> = (1..=5).map(|i| read_a(&format!("c{i}"))).collect();
        script.push(text("done"));
        let mut m = Scripted::new(script);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("repeat", Mode::Writes, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("read"), Outcome::Done);
        let session = a.session;

        assert_eq!(nudges(&session.events, "repeat"), ["this call returned the same result 3 times — change approach"]);
        assert_eq!(last_user(&m.requests[3]), "this call returned the same result 3 times — change approach");
    }

    #[test]
    fn empty_response_nudge_then_error() {
        let mut m = Scripted::new(vec![text(""), text(""), text("never")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("empty", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("go"), Outcome::Error("empty response".into()));
        let session = a.session;

        assert_eq!(nudges(&session.events, "empty"), ["(empty response — continue the task, or reply with your answer)"]);
        assert_eq!(m.requests.len(), 2);
        assert_eq!(ui.notices, ["empty response"]);
        assert_eq!(ui.statuses.len(), 1);
    }

    #[test]
    fn step_limit() {
        let todo = |i: usize| call(&format!("c{i}"), "todo", &json!({ "items": [{ "text": format!("t{i}"), "status": "pending" }] }).to_string());
        let mut m = Scripted::new((0..101).map(todo).collect());
        let (mut ui, cancel) = (TestUi { confirms: [false].into(), ..Default::default() }, AtomicBool::new(false));
        let mut a = agent("steps", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("plan"), Outcome::StepLimit);
        drop(a);
        assert_eq!(m.requests.len(), 100);
        assert_eq!(ui.questions, ["step limit reached (100) — continue?"]);

        // "yes" starts a fresh count of 100
        let mut script: Vec<_> = (0..100).map(todo).collect();
        script.push(text("done"));
        let mut m = Scripted::new(script);
        let mut ui = TestUi { confirms: [true].into(), ..Default::default() };
        let mut a = agent("steps", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("plan"), Outcome::Done);
        drop(a);
        assert_eq!((m.requests.len(), ui.questions.len()), (101, 1));
    }

    #[test]
    fn big_output_goes_to_artifact() {
        let mut m = Scripted::new(vec![read_a("c1"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("big", Mode::Writes, false, &mut m, &mut ui, &cancel);
        let line = "x".repeat(100);
        fs::write(a.tools.root.join("a.txt"), vec![line.as_str(); 1_500].join("\n")).unwrap();
        assert_eq!(a.run_request("read it"), Outcome::Done);
        let session = a.session;

        let (output, artifact, bytes) = session
            .events
            .iter()
            .find_map(|e| match &e.kind {
                Kind::ToolResult { output, artifact, bytes, .. } => Some((output.clone(), artifact.clone(), *bytes)),
                _ => None,
            })
            .unwrap();
        let artifact = artifact.expect("a truncated result has an artifact");
        assert_eq!(Path::new(&artifact), session.artifacts_dir().join("c1.txt"));
        let full = fs::read_to_string(&artifact).unwrap();
        assert_eq!(full.len() as u64, bytes);
        assert!(full.ends_with(&format!("  1500\t{line}")), "the artifact holds the whole output");
        assert_eq!(output, format!("{}{}", tools::clip(&full, policy::LIMITS.tool_cap), truncation_note(full.len(), Path::new(&artifact))));
        assert!(output.ends_with(" — use read/grep]"));
    }

    #[test]
    fn truncation_note_format() {
        assert_eq!(truncation_note(1, Path::new("/s/artifacts/c1.txt")), "\n[truncated: full output 1KB at /s/artifacts/c1.txt — use read/grep]");
        assert_eq!(truncation_note(2048, Path::new("p")), "\n[truncated: full output 2KB at p — use read/grep]");
        assert_eq!(truncation_note(2049, Path::new("p")), "\n[truncated: full output 3KB at p — use read/grep]");
    }

    #[test]
    fn llm_error_recorded() {
        let mut m = Scripted::new(vec![Err(LlmError::Network("reset".into()))]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("llmerr", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("go"), Outcome::Error("network error: reset".into()));
        let session = a.session;

        assert_eq!(session.events.last().unwrap().kind, Kind::Error { kind: "llm".into(), message: "network error: reset".into(), retry: false });
        assert_eq!(ui.notices, ["network error: reset"]);
        assert_eq!(ui.statuses.len(), 1);
    }

    #[test]
    fn auth_error_message() {
        let mut m = Scripted::new(vec![Err(LlmError::Auth("HTTP 401: no".into()))]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("auth", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("go"), Outcome::Error("authentication failed — check ZAI_API_KEY".into()));
    }

    #[test]
    fn every_event_reaches_ui() {
        let mut m = Scripted::new(vec![read_a("c1"), call("c2", "edit", r#"{"path":"a.txt","old":"hello","new":"bye"}"#), text("done"), text("verified")]);
        let (mut ui, cancel) = (TestUi { decisions: [Decision::Allow].into(), ..Default::default() }, AtomicBool::new(false));
        let mut a = agent("allui", Mode::Writes, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("change it"), Outcome::Done);
        let session = a.session;

        // everything after the helper's SessionStart was appended by the agent, and each one reached the Ui
        assert_eq!(ui.events, session.events[1..]);
        assert_eq!(ui.events.len(), 9); // user, 2 × (assistant, result), permission, assistant, nudge, assistant
    }
}

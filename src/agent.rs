//! The turn loop (spec §11, §12): for each user request, build the view from the events, stream the model call,
//! run tool calls through permissions, record every step as an event, and apply the loop guards.

use crate::llm::{Delta, LlmError, Message, Model, Reply, ToolCall};
use crate::perms::{Perms, Verdict};
use crate::policy::{self, Compaction, LIMITS};
use crate::store::{Kind, Session};
use crate::tools::{self, ToolCtx};
use crate::ui::{Decision, PermissionRequest, Status, Ui, mini_diff, tool_summary};
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How an `Outcome::Error` (or a `/compact` error) from a failed session write starts. After one, the events file may end
/// in a partial line, so nothing more may be appended to that session.
pub const SESSION_WRITE_FAILED: &str = "cannot write session";
const CANCELLED: &str = "cancelled by user";
const HEADLESS_DENIED: &str = "denied: not allowed in headless mode (use --mode or --allow)";
const EMPTY_NUDGE: &str = "(empty response — continue the task, or reply with your answer)";
/// The text of an assistant turn the thinking cap ended (like `[interrupted]` for a Ctrl+C): keeps the turn non-empty for the API.
const THINK_CUT: &str = "[thinking cut off]";
/// Lines of an edit/write diff shown in a permission prompt.
const DIFF_LINES: usize = 20;
/// Tool output longer than this (bytes) is also saved to `artifacts/<call-id>.txt`, so its stage-1 stub has a `full:` path.
const ARTIFACT_MIN: usize = 1536;

pub struct Agent<'a> {
    pub session: Session,
    pub model: &'a mut dyn Model,
    pub ui: &'a mut dyn Ui,
    pub perms: Perms,
    pub tools: ToolCtx,
    pub budget: usize,
    pub cancel: &'a AtomicBool,
    /// Named in the auth error (spec §12: "check <api_key_env>").
    pub api_key_env: &'a str,
    /// `--deadline`: when set, the effort cap and `time` nudges follow the time left (policy.rs).
    pub deadline: Option<Deadline>,
}

/// The moment a run will be killed, and the budget it started with.
#[derive(Clone, Copy)]
pub struct Deadline {
    end: Instant,
    total: Duration,
}

impl Deadline {
    pub fn new(start: Instant, total: Duration) -> Self {
        Deadline { end: start + total, total }
    }

    fn remaining(&self) -> Duration {
        self.end.saturating_duration_since(Instant::now())
    }
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

    /// Stage-2 compaction for `/compact [focus]` (spec §10.4). A failure is recorded as an `error` event and leaves
    /// the view unchanged; its message comes back for the caller to show.
    pub fn compact(&mut self, focus: Option<&str>) -> Result<(), String> {
        self.cancel.store(false, Ordering::SeqCst); // a Ctrl+C from an earlier request must not cut the summary call
        match self.stage2(focus, false) {
            Ok(result) => result,
            Err(Outcome::Error(fatal)) => Err(fatal),
            Err(other) => unreachable!("a failed session write is always Outcome::Error, got {other:?}"),
        }
    }

    /// The compaction triggers (spec §10.2–10.4), checked before every model call. Once a stage 2 fails, `stage2_failed`
    /// keeps the rest of the request from paying for another summary call at every step (stage 1 may still run).
    fn compact_if_needed(&mut self, stage2_failed: &mut bool) -> Result<(), Outcome> {
        let tokens = policy::estimate_tokens(&self.session.events);
        match policy::compaction_needed(tokens, self.budget, &self.session.events) {
            None => {}
            Some(Compaction::Stage2) if *stage2_failed => {}
            Some(Compaction::Stage1 { upto_seq }) => {
                let kind = Kind::Compaction {
                    stage: 1,
                    upto_seq: Some(upto_seq),
                    from_seq: None,
                    to_seq: None,
                    summary_path: None,
                    summary: None,
                    tokens_before: None,
                };
                self.append(kind)?;
                self.ui.notice(&format!("compacted: elided old tool output (≤ seq {upto_seq})"));
            }
            Some(Compaction::Stage2) => {
                if let Err(e) = self.stage2(None, false)? {
                    *stage2_failed = true;
                    self.ui.notice(&e); // the loop goes on with the view unchanged
                }
            }
        }
        Ok(())
    }

    /// Stage 2 (spec §10.4). `Ok(Err(msg))`: the compaction failed; it is recorded as an `error` event and the view is
    /// unchanged. `Err`: a failed session write, as with `append`. `overflowed`: the API rejected the current view as too
    /// long, so the summary call sends it with every tool result stubbed.
    fn stage2(&mut self, focus: Option<&str>, overflowed: bool) -> Result<Result<(), String>, Outcome> {
        let events = &self.session.events;
        let Some(cut) = policy::stage2_cut(events, self.budget) else {
            self.ui.notice("nothing to compact");
            return Ok(Ok(()));
        };
        let tokens_before = policy::estimate_tokens(events) as u64;
        self.model.think_cap(None); // a summary is not paced: a cut one would only fail the compaction
        let mut view = if overflowed { policy::stubbed_view(events) } else { policy::build_view(events) };
        let prompt = policy::summary_prompt(focus);
        match view.last_mut() {
            // the view never holds two user messages in a row
            Some(Message::User(last)) => {
                last.push_str("\n\n");
                last.push_str(&prompt);
            }
            _ => view.push(Message::User(prompt)),
        }
        self.ui.notice(&format!("compacting… (summarizing events {}–{})", cut.from_seq, cut.to_seq));
        // Tools disabled: the API accepts a history with tool calls without them (spec §16 item 8). The summary is not an
        // answer, so only notices reach the Ui.
        let ui = &mut *self.ui;
        let reply = self.model.complete(
            &view,
            &[],
            &mut |d| {
                if let Delta::Notice(n) = d {
                    ui.notice(&n)
                }
            },
            self.cancel,
        );
        let written = match reply {
            Err(e) => Err(e.to_string()),
            Ok(r) if r.interrupted => Err("interrupted".to_string()),
            Ok(r) if r.text.trim().is_empty() => Err("empty summary".to_string()),
            Ok(r) => {
                let summary = policy::summary_message(&r.text, &cut, &self.session.events, &self.session.dir);
                self.session.write_summary(&summary).map(|path| (summary, path)).map_err(|e| format!("cannot write summary: {e}"))
            }
        };
        let (summary, path) = match written {
            Ok(w) => w,
            Err(e) => {
                let message = format!("compaction failed: {e}");
                self.append(Kind::Error { kind: "compaction".into(), message: message.clone(), retry: false })?;
                return Ok(Err(message));
            }
        };
        let (from_seq, to_seq, summary_path) = (cut.from_seq, cut.to_seq, path.display().to_string());
        self.append(Kind::Compaction {
            stage: 2,
            upto_seq: None,
            from_seq: Some(from_seq),
            to_seq: Some(to_seq),
            summary_path: Some(summary_path.clone()),
            summary: Some(summary),
            tokens_before: Some(tokens_before),
        })?;
        self.ui.notice(&format!("compacted: summary of events {from_seq}–{to_seq} → {summary_path}"));
        Ok(Ok(()))
    }

    /// The loop of spec §11. `Err` is a failed session write, which ends the request at once: nothing may be
    /// appended after a partial write.
    fn turn(&mut self, request: &str, tool_calls: &mut usize) -> Result<Outcome, Outcome> {
        self.cancel.store(false, Ordering::SeqCst);
        self.append(Kind::User { text: request.into() })?;
        let schemas = tools::schemas();
        let (mut steps, mut check_fired, mut edited, mut empty_retried, mut overflow_retried) = (0, false, false, false, false);
        let mut recent: Vec<Seen> = Vec::new(); // tool results since the last repeat nudge
        let mut stage2_failed = false;
        let mut time_notes = 0;
        let (mut cuts, mut after_cut) = (0, false); // thinking-cap cuts so far, and whether the next call follows one
        loop {
            // Before every model call, a request's first included: tool-free turns grow the context too. Not while a
            // Ctrl+C is pending: the check below ends the request, and a summary call now would only be cut short.
            if !self.cancel.load(Ordering::SeqCst) {
                self.compact_if_needed(&mut stage2_failed)?;
            }
            // Ctrl+C during the last tool of a batch (or at its prompt, or during a summary call) must not send another model request
            if self.cancel.load(Ordering::SeqCst) {
                return self.interrupt(&[], "tool", None);
            }
            if steps == LIMITS.step_limit {
                if !self.ui.confirm(&format!("step limit reached ({}) — continue?", LIMITS.step_limit)) {
                    return Ok(Outcome::StepLimit);
                }
                steps = 0;
            }
            self.pace(&mut time_notes, cuts, std::mem::take(&mut after_cut))?;
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
                Err(e @ LlmError::ContextOverflow(_)) if !overflow_retried => {
                    overflow_retried = true;
                    // the trace records why the history is compacted; then the request is retried once
                    self.append(Kind::Error { kind: "llm".into(), message: e.to_string(), retry: true })?;
                    if let Err(e) = self.stage2(None, true)? {
                        self.ui.notice(&e);
                        return Ok(Outcome::Error(e));
                    }
                    continue;
                }
                Err(e) => {
                    let msg = match e {
                        LlmError::Auth(_) => format!("authentication failed — check {}", self.api_key_env),
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
            let Reply { text, reasoning, tool_calls: calls, usage, finish_reason, interrupted, capped, think_ms, effort } = reply;
            if interrupted {
                let text = format!("{text}\n[interrupted]");
                self.append(Kind::Assistant { text, reasoning, tool_calls: calls.clone(), usage, finish_reason, latency_ms, effort, think_ms })?;
                return self.interrupt(&calls, "stream", None);
            }
            if capped {
                // The thought stays in the history, as for an interrupt, so the model acts on it. The turn carries a marker as its
                // text and no tool calls whatever the reply held, so nothing is left unanswered; a nudge follows, and the next
                // call runs at `low`.
                let thought = Duration::from_millis(think_ms.unwrap_or(0));
                let finish_reason = "think_cap".to_string();
                self.append(Kind::Assistant { text: THINK_CUT.into(), reasoning, tool_calls: vec![], usage, finish_reason, latency_ms, effort, think_ms })?;
                self.append(nudge("think_cap", policy::think_note(thought)))?;
                (cuts, after_cut) = (cuts + 1, true);
                continue;
            }
            self.append(Kind::Assistant { text: text.clone(), reasoning, tool_calls: calls.clone(), usage, finish_reason, latency_ms, effort, think_ms })?;

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

    /// With a deadline, before every model call: caps the reasoning effort by the time left (at `low` right after a thinking
    /// cut), bounds how long the call may reason before it acts (until `think_cap_fires` cuts, `cuts`, have been used), and
    /// the first time the budget falls to each `time_note_pct` threshold adds one `time` nudge. `notes`: how many
    /// thresholds have fired. Without a deadline nothing is set, so the model runs exactly as before.
    fn pace(&mut self, notes: &mut usize, cuts: usize, after_cut: bool) -> Result<(), Outcome> {
        let Some(d) = self.deadline else { return Ok(()) };
        let left = d.remaining();
        self.model.cap_effort(if after_cut { Some("low") } else { policy::effort_cap(left) });
        self.model.think_cap((cuts < LIMITS.think_cap_fires).then(|| policy::think_cap(left)));
        let pct = (left.as_secs_f64() * 100.0 / d.total.as_secs_f64()) as u64;
        let crossed = LIMITS.time_note_pct.iter().filter(|&&p| pct <= p).count();
        if crossed > *notes {
            *notes = crossed;
            self.append(nudge("time", policy::time_note(crossed - 1, left)))?;
        }
        Ok(())
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
            Verdict::Ask => Some(match self.ui.ask(&permission_request(call, &args, self.perms.grant_preview(&call.name, &args))) {
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
        // A stage-1 stub can only point at the full text if it is saved: following `events.jsonl seq N` instead gives
        // lines cut by `read`/`grep`. So every result past ARTIFACT_MIN is saved, not only a truncated one.
        let saved = (capped.truncated || out.text.len() > ARTIFACT_MIN).then(|| self.session.write_artifact(&format!("{}.txt", call.id), out.text.as_bytes()));
        let (output, artifact) = match saved {
            None => (capped.text, None),
            Some(Ok(path)) if capped.truncated => (capped.text + &truncation_note(out.text.len(), &path), Some(path.display().to_string())),
            Some(Ok(path)) => (capped.text, Some(path.display().to_string())),
            Some(Err(e)) if capped.truncated => (format!("{}\n[truncated: the full output could not be saved: {e}]", capped.text), None),
            Some(Err(_)) => (capped.text, None), // the result itself is whole; only its stub will lack a `full:` path
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
        let e = self.session.append(kind).map_err(|e| Outcome::Error(format!("{SESSION_WRITE_FAILED}: {e}")))?;
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
/// what they approve), and `edit`/`write` show a diff. `always` is what an "always" answer grants.
fn permission_request(call: &ToolCall, args: &Value, always: String) -> PermissionRequest {
    let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("");
    let detail = match call.name.as_str() {
        "bash" => Some(s("command").to_string()),
        "edit" => Some(mini_diff(s("old"), s("new"), DIFF_LINES)),
        "write" => Some(mini_diff("", s("content"), DIFF_LINES)),
        _ => None,
    };
    PermissionRequest { tool: call.name.clone(), summary: tool_summary(call), detail, always }
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
        let tools = ToolCtx { root: proj, shell: tools::resolve_shell(), backups_dir: session.artifacts_dir(), reads: HashMap::new(), secret_env: vec![] };
        let perms = Perms::new(mode, headless, session.dir.clone(), vec![]);
        Agent { session, model, ui, perms, tools, budget: 200_000, cancel, api_key_env: "XOLOTL_TEST_KEY", deadline: None }
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
    /// A read of `f{i}.txt` (a distinct file each time, so the repeat guard stays quiet) whose reply reports `prompt` tokens.
    fn read_f(i: usize, prompt: u64) -> Result<Reply, LlmError> {
        let reply = call(&format!("c{i}"), "read", &format!(r#"{{"path":"f{i}.txt"}}"#)).unwrap();
        Ok(Reply { usage: Usage { prompt, ..Usage::default() }, ..reply })
    }
    fn write_files(root: &Path, n: usize, content: &str) {
        for i in 1..=n {
            fs::write(root.join(format!("f{i}.txt")), content).unwrap();
        }
    }
    fn compactions(events: &[Event]) -> Vec<&Event> {
        events.iter().filter(|e| matches!(e.kind, Kind::Compaction { .. })).collect()
    }
    fn stages(events: &[Event]) -> Vec<u8> {
        compactions(events).iter().filter_map(|e| if let Kind::Compaction { stage, .. } = e.kind { Some(stage) } else { None }).collect()
    }
    /// `(message, retry)` of every error event of `kind`.
    fn errors(events: &[Event], kind: &str) -> Vec<(String, bool)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::Error { kind: k, message, retry } if k == kind => Some((message.clone(), *retry)),
                _ => None,
            })
            .collect()
    }
    /// A finished earlier request (R6), so a stage-2 cut has something to cover.
    fn seed_earlier_turn(a: &mut Agent<'_>) {
        a.session.append(Kind::User { text: "earlier".into() }).unwrap();
        let reply = Kind::Assistant {
            text: "ok".into(),
            reasoning: String::new(),
            tool_calls: vec![],
            usage: Usage::default(),
            finish_reason: "stop".into(),
            latency_ms: 0,
            effort: None,
            think_ms: None,
        };
        a.session.append(reply).unwrap();
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

    fn deadline_in(left_s: u64, total_s: u64) -> Deadline {
        Deadline { end: Instant::now() + Duration::from_secs(left_s), total: Duration::from_secs(total_s) }
    }

    #[test]
    fn without_a_deadline_nothing_is_paced() {
        let mut m = Scripted::new(vec![read_a("c1"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("nopace", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert!(m.caps.is_empty() && m.think_caps.is_empty() && nudges(&session.events, "time").is_empty());
    }

    #[test]
    fn deadline_caps_effort_and_notes_each_threshold_once() {
        // 400 s of 1000 s left: past the 50% threshold, under the 1000 s effort mark, over the 240 s one
        let mut m = Scripted::new(vec![read_a("c1"), read_a("c2"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("pace", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(400, 1000));
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(m.caps, [Some("medium".to_string()), Some("medium".into()), Some("medium".into())]); // one per model call
        let notes = nudges(&session.events, "time");
        assert_eq!(notes.len(), 1, "three calls, one note: {notes:?}");
        assert!(notes[0].contains("about 7 minutes") && notes[0].contains("exists and works now"), "{}", notes[0]);
        // the note sits before the first call, merged into the request's user message, so the model sees it at once
        let first = last_user(&m.requests[0]);
        assert!(first.starts_with("look") && first.contains("Time check"), "{first}");
    }

    #[test]
    fn deadline_far_past_both_thresholds_adds_only_the_last_note() {
        let mut m = Scripted::new(vec![read_a("c1"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("pace2", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(100, 1000)); // 10% left: both thresholds crossed at once
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(m.caps, [Some("low".to_string()), Some("low".into())]);
        let notes = nudges(&session.events, "time");
        assert!(notes.len() == 1 && notes[0].contains("Wrap up"), "{notes:?}");
    }

    #[test]
    fn plenty_of_time_means_no_cap_and_no_note() {
        let mut m = Scripted::new(vec![text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("pace3", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(3000, 3600));
        assert_eq!(a.run_request("hi"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(m.caps, [None]); // the cap is lifted explicitly, so an earlier one cannot stick
        assert!(nudges(&session.events, "time").is_empty());
    }

    /// `(effort, think_ms)` of every assistant event.
    fn trace_fields(events: &[Event]) -> Vec<(Option<String>, Option<u64>)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::Assistant { effort, think_ms, .. } => Some((effort.clone(), *think_ms)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn assistant_events_record_the_effort_sent_and_the_think_time() {
        let thought = Reply { reasoning: "hmm".into(), think_ms: Some(4_200), ..read_a("c1").unwrap() };
        let mut m = Scripted::new(vec![Ok(thought), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("tracefields", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(400, 1000)); // medium
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(trace_fields(&session.events), [(Some("medium".to_string()), Some(4_200)), (Some("medium".to_string()), None)]);

        // no deadline: nothing is sent, so nothing is recorded as sent
        let mut m = Scripted::new(vec![text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("tracefields2", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.run_request("hi"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(trace_fields(&session.events), [(None, None)]);
    }

    /// What a client returns when its thinking cap ended the call: reasoning only.
    fn capped(thought: &str, think_ms: u64) -> Result<Reply, LlmError> {
        Ok(Reply { reasoning: thought.into(), capped: true, think_ms: Some(think_ms), ..Default::default() })
    }
    /// A request's messages must never hold an assistant turn with neither text nor tool calls.
    fn no_empty_assistant(req: &(Vec<Message>, usize)) -> bool {
        req.0.iter().all(|m| !matches!(m, Message::Assistant { text, tool_calls, .. } if text.is_empty() && tool_calls.is_empty()))
    }

    #[test]
    fn thinking_cap_follows_the_time_left_and_needs_a_deadline() {
        let s = Duration::from_secs;
        let cap_for = |tag: &str, left_s: Option<u64>| {
            let mut m = Scripted::new(vec![text("done")]);
            let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
            let mut a = agent(tag, Mode::Yolo, false, &mut m, &mut ui, &cancel);
            a.deadline = left_s.map(|n| deadline_in(n, 3600));
            assert_eq!(a.run_request("hi"), Outcome::Done);
            m.think_caps
        };
        let caps = cap_for("thinkcap1", Some(2000)); // half the time left, as the clock has moved a little
        assert!(caps.len() == 1 && caps[0].is_some_and(|c| c <= s(1000) && c > s(990)), "{caps:?}");
        assert_eq!(cap_for("thinkcap2", Some(100)), [Some(s(150))]); // the floor
        assert_eq!(cap_for("thinkcap3", None), []); // no deadline: no cap is ever set
    }

    #[test]
    fn a_cut_thought_is_kept_and_nudged_and_the_request_goes_on() {
        let mut m = Scripted::new(vec![capped("long thought", 420_000), read_a("c1"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("cut1", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(3000, 3600));
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;

        // the trace: the turn with its reasoning and think time, the nudge right after it, and no interrupt
        let at = session.events.iter().position(|e| matches!(&e.kind, Kind::Assistant { finish_reason, .. } if finish_reason == "think_cap")).unwrap();
        let Kind::Assistant { text, reasoning, tool_calls, think_ms, .. } = &session.events[at].kind else { unreachable!() };
        assert_eq!((text.as_str(), reasoning.as_str(), tool_calls.len(), *think_ms), ("[thinking cut off]", "long thought", 0, Some(420_000)));
        let Kind::Nudge { reason, text: note } = &session.events[at + 1].kind else { panic!("no nudge after the cut") };
        assert_eq!(reason, "think_cap");
        assert!(note.contains("about 7 minutes") && note.contains("simplest working version"), "{note}");
        assert!(interrupts(&session.events).is_empty());

        // the request: the thought is resent with the turn, then the note as the next user message; nothing the API would refuse
        let second = &m.requests[1];
        let n = second.0.len();
        assert!(matches!(&second.0[n - 2], Message::Assistant { text, reasoning: Some(r), tool_calls } if text == "[thinking cut off]" && r == "long thought" && tool_calls.is_empty()));
        let wire = second.0[n - 2].to_wire(); // as the API receives it: non-empty content, the thought, no tool_calls key
        assert_eq!((wire["content"].as_str(), wire["reasoning_content"].as_str(), wire.get("tool_calls")), (Some("[thinking cut off]"), Some("long thought"), None));
        assert_eq!(last_user(second), *note);
        assert!(m.requests.iter().all(no_empty_assistant));
        assert_eq!(m.requests.len(), 3); // the cut call, the call after it, the call after the read
    }

    #[test]
    fn a_cut_turn_never_leaves_tool_calls_unanswered() {
        // a client never returns a capped reply with calls, but the turn must not depend on it
        let stray = Ok(Reply { tool_calls: vec![tc("c9", "read", "{}")], text: "x".into(), ..capped("t", 1).unwrap() });
        let mut m = Scripted::new(vec![stray, text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("cut2", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(3000, 3600));
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert!(results(&session.events).is_empty() && ui.started.is_empty());
        assert!(m.requests.iter().all(|r| r.0.iter().all(|msg| !matches!(msg, Message::Assistant { tool_calls, .. } if !tool_calls.is_empty()))));
    }

    #[test]
    fn the_call_after_a_cut_runs_at_low_and_only_that_one() {
        let mut m = Scripted::new(vec![capped("t", 1), read_a("c1"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("cut3", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(3000, 3600)); // plenty of time: no time cap of its own
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(m.caps, [None, Some("low".to_string()), None]);
        let efforts: Vec<_> = trace_fields(&session.events).into_iter().map(|(e, _)| e).collect();
        assert_eq!(efforts, [None, Some("low".to_string()), None]); // and the trace says so

        // under a time cap the call after a cut is still `low`, then the time cap applies again
        let mut m = Scripted::new(vec![capped("t", 1), read_a("c1"), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("cut3b", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(900, 1000));
        fs::write(a.tools.root.join("a.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("look"), Outcome::Done);
        assert_eq!(m.caps, [Some("high".to_string()), Some("low".into()), Some("high".into())]);
    }

    #[test]
    fn the_cap_stops_after_its_cuts_are_used_up() {
        let mut m = Scripted::new(vec![capped("a", 1), capped("b", 1), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("cut4", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(3000, 3600));
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        assert_eq!(nudges(&session.events, "think_cap").len(), LIMITS.think_cap_fires);
        let on: Vec<bool> = m.think_caps.iter().map(Option::is_some).collect();
        assert_eq!(on, [true, true, false]); // the third call is not capped
    }

    #[test]
    fn the_cut_and_the_time_note_keep_their_order() {
        // 10% of the budget left: the time note is added before the first call, the cut note right after the cut
        let mut m = Scripted::new(vec![capped("t", 150_000), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("cut5", Mode::Yolo, false, &mut m, &mut ui, &cancel);
        a.deadline = Some(deadline_in(100, 1000));
        assert_eq!(a.run_request("look"), Outcome::Done);
        let Agent { session, .. } = a;
        let kinds: Vec<&str> = session.events.iter().filter_map(|e| match &e.kind {
            Kind::Nudge { reason, .. } => Some(reason.as_str()),
            Kind::Assistant { .. } => Some("assistant"),
            _ => None,
        }).collect();
        assert_eq!(kinds, ["time", "assistant", "think_cap", "assistant"]);
        assert_eq!(m.caps, [Some("low".to_string()), Some("low".into())]);
        assert_eq!(last_user(&m.requests[1]), policy::think_note(Duration::from_secs(150)));
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
        assert_eq!(req.always, "edit");
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
        assert_eq!(ui.asks[0].always, "echo hi, rm -rf"); // every line is a segment, so "always" would grant both
        assert_eq!(ui.asks[1].always, "write");
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
    fn mid_size_output_gets_an_artifact() {
        // a 2 KB read (under the 30 KB cap) and a small one
        let mut m = Scripted::new(vec![calls(vec![tc("c1", "read", r#"{"path":"a.txt"}"#), tc("c2", "read", r#"{"path":"b.txt"}"#)]), text("done")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("midsize", Mode::Writes, false, &mut m, &mut ui, &cancel);
        fs::write(a.tools.root.join("a.txt"), "x".repeat(2_000)).unwrap();
        fs::write(a.tools.root.join("b.txt"), "hello\n").unwrap();
        assert_eq!(a.run_request("read them"), Outcome::Done);
        let session = a.session;

        let found: Vec<(u64, String, Option<String>)> = session
            .events
            .iter()
            .filter_map(|e| match &e.kind {
                Kind::ToolResult { output, artifact, .. } => Some((e.seq, output.clone(), artifact.clone())),
                _ => None,
            })
            .collect();
        let (seq, output, artifact) = &found[0];
        let artifact = artifact.as_ref().expect("a result over 1.5 KB has an artifact");
        assert_eq!(Path::new(artifact), session.artifacts_dir().join("c1.txt"));
        assert_eq!(&fs::read_to_string(artifact).unwrap(), output, "not truncated: the artifact is the whole output");
        assert!(!output.contains("[truncated"), "only truncated output gets the note");
        assert_eq!(found[1].2, None, "a small result has none");
        // so its stage-1 stub points at the full text
        let stubs: Vec<String> = policy::render(&session.events, 0, *seq)
            .into_iter()
            .filter_map(|m| if let Message::Tool { call_id, content } = m { (call_id == "c1").then_some(content) } else { None })
            .collect();
        assert_eq!(stubs, [format!("[elided: 2KB read output — events.jsonl seq {seq}; full: {artifact}]")]);
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
        // the configured variable's name, not the default's
        assert_eq!(a.run_request("go"), Outcome::Error("authentication failed — check XOLOTL_TEST_KEY".into()));
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

    // ---- compaction (Task 12) ----

    #[test]
    fn stage1_fires_and_stubs() {
        // budget 3_000: 11 small reads, a 12th whose reply reports 1_900 tokens (≥ 60%), a 13th back at 1_000, then the answer
        let mut script: Vec<_> = (1..=11).map(|i| read_f(i, 100)).collect();
        script.extend([read_f(12, 1_900), read_f(13, 1_000), text("done")]);
        let mut m = Scripted::new(script);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("stage1", Mode::Writes, false, &mut m, &mut ui, &cancel);
        a.budget = 3_000;
        write_files(&a.tools.root, 13, "x");
        assert_eq!(a.run_request("read them"), Outcome::Done);
        let session = a.session;

        let result_seqs: Vec<u64> = session.events.iter().filter(|e| matches!(e.kind, Kind::ToolResult { .. })).map(|e| e.seq).collect();
        let upto = result_seqs[1]; // the 11th-most-recent of 12 results
        let c = compactions(&session.events);
        assert_eq!(c.len(), 1);
        assert_eq!(
            c[0].kind,
            Kind::Compaction { stage: 1, upto_seq: Some(upto), from_seq: None, to_seq: None, summary_path: None, summary: None, tokens_before: None }
        );
        assert_eq!(ui.notices, [format!("compacted: elided old tool output (≤ seq {upto})")]);
        assert_eq!(m.requests.len(), 14); // no summary call
        // the request after it shows results 1 and 2 as stubs, the other ten verbatim
        let tool_texts: Vec<&str> =
            m.requests[12].0.iter().filter_map(|m| if let Message::Tool { content, .. } = m { Some(content.as_str()) } else { None }).collect();
        assert_eq!(tool_texts.len(), 12);
        assert!(tool_texts[..2].iter().all(|t| t.starts_with("[elided:")), "{tool_texts:?}");
        assert!(tool_texts[2..].iter().all(|t| !t.contains("[elided:")), "{tool_texts:?}");
        assert!(m.requests[11].0.iter().all(|m| !matches!(m, Message::Tool { content, .. } if content.contains("[elided:"))));
    }

    #[test]
    fn stage2_after_failed_stage1() {
        // as above, but the 13th reply is still at 1_900 (≥ 60% right after stage 1): stage 2, whose summary is "SUM"
        let mut script: Vec<_> = (1..=11).map(|i| read_f(i, 100)).collect();
        script.extend([read_f(12, 1_900), read_f(13, 1_900), text("SUM"), text("done")]);
        let mut m = Scripted::new(script);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("stage2", Mode::Writes, false, &mut m, &mut ui, &cancel);
        a.budget = 3_000;
        write_files(&a.tools.root, 13, "x");
        assert_eq!(a.run_request("read them"), Outcome::Done);
        let session = a.session;

        assert_eq!(m.requests.len(), 15);
        let (summary_view, tool_count) = &m.requests[13];
        assert_eq!(*tool_count, 0); // tools disabled
        assert_eq!(summary_view.last(), Some(&Message::User(policy::summary_prompt(None))));
        assert_eq!(stages(&session.events), [1, 2]);
        let c = compactions(&session.events)[1];
        let Kind::Compaction { from_seq: Some(from), to_seq: Some(to), summary_path: Some(path), summary: Some(summary), tokens_before: Some(tokens), .. } =
            &c.kind
        else {
            panic!("incomplete stage-2 event: {c:?}")
        };
        assert!(summary.contains("SUM") && summary.contains("## Table of contents"), "{summary}");
        assert_eq!(Path::new(path), session.dir.join("summaries").join("1.md"));
        assert_eq!(&fs::read_to_string(path).unwrap(), summary);
        // the size measured just before the compaction event was appended
        assert_eq!(*tokens, policy::estimate_tokens(&session.events[..c.seq as usize - 1]) as u64);
        assert_eq!(ui.notices.last().unwrap(), &format!("compacted: summary of events {from}–{to} → {path}"));
        assert_eq!(ui.text, "done"); // the summary is not streamed as an answer
        // the next request starts from the summary
        let next = &m.requests[14].0;
        assert_eq!(next[0], Message::System("S".into()));
        assert_eq!(next[1], Message::User(summary.clone()));
    }

    #[test]
    fn manual_compact_with_focus() {
        let mut m = Scripted::new(vec![text("SUM")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(true)); // a Ctrl+C from an earlier request must not cut the summary
        let mut a = agent("focus", Mode::Writes, false, &mut m, &mut ui, &cancel);
        seed_earlier_turn(&mut a);
        assert_eq!(a.compact(Some("parser")), Ok(()));
        let session = a.session;

        assert!(!cancel.load(Ordering::SeqCst));
        assert_eq!(m.requests.len(), 1);
        assert_eq!(m.requests[0].1, 0);
        assert!(last_user(&m.requests[0]).ends_with("Do not call tools.\nFocus: parser"));
        assert_eq!(stages(&session.events), [2]);
        assert_eq!(session.events.len(), 4); // SessionStart, the seeded turn, the compaction
    }

    #[test]
    fn overflow_compacts_and_retries_once() {
        let overflow = || Err(LlmError::ContextOverflow("prompt too long".into()));
        let mut m = Scripted::new(vec![overflow(), text("SUM"), text("ok")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("overflow", Mode::Writes, false, &mut m, &mut ui, &cancel);
        seed_earlier_turn(&mut a);
        assert_eq!(a.run_request("go"), Outcome::Done);
        let session = a.session;

        assert_eq!(m.requests.len(), 3);
        // the trace records why it compacted
        assert_eq!(errors(&session.events, "llm"), [("context window exceeded: prompt too long".to_string(), true)]);
        assert_eq!(stages(&session.events), [2]);
        // the summary call: the view ends with the request, so the prompt joins it rather than adding a second user message
        assert_eq!(m.requests[1].1, 0);
        assert_eq!(last_user(&m.requests[1]), format!("go\n\n{}", policy::summary_prompt(None)));
        // the retry starts from the summary (the cut fell just before "go", which merges into it)
        let retry = &m.requests[2].0;
        assert!(matches!(&retry[..], [Message::System(_), Message::User(u)] if u.starts_with("[Session summary") && u.ends_with("\n\ngo")), "{retry:?}");

        // only once: a second overflow ends the request
        let mut m = Scripted::new(vec![overflow(), text("SUM"), overflow(), text("never")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("overflow2", Mode::Writes, false, &mut m, &mut ui, &cancel);
        seed_earlier_turn(&mut a);
        assert_eq!(a.run_request("go"), Outcome::Error("context window exceeded: prompt too long".into()));
        let session = a.session;

        assert_eq!(m.requests.len(), 3);
        assert_eq!(stages(&session.events), [2]);
        let e = errors(&session.events, "llm");
        assert_eq!(e.iter().map(|(_, retry)| *retry).collect::<Vec<_>>(), [true, false]);
    }

    #[test]
    fn overflow_summary_is_sent_with_results_stubbed() {
        let mut m = Scripted::new(vec![Err(LlmError::ContextOverflow("prompt too long".into())), text("SUM"), text("ok")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("overflowstub", Mode::Writes, false, &mut m, &mut ui, &cancel);
        // an earlier turn whose tool result is big (seq 2..=5)
        let reply = |text: &str, tool_calls: Vec<ToolCall>| Kind::Assistant {
            text: text.into(),
            reasoning: String::new(),
            tool_calls,
            usage: Usage::default(),
            finish_reason: "stop".into(),
            latency_ms: 0,
            effort: None,
            think_ms: None,
        };
        let big = "r".repeat(5_000);
        a.session.append(Kind::User { text: "earlier".into() }).unwrap();
        a.session.append(reply("", vec![tc("c0", "bash", "{}")])).unwrap();
        let output = Kind::ToolResult { call_id: "c0".into(), name: "bash".into(), ok: true, output: big.clone(), artifact: None, bytes: 5_000, duration_ms: 0, exit_code: Some(0) };
        a.session.append(output).unwrap();
        a.session.append(reply("ok", vec![])).unwrap();
        assert_eq!(a.run_request("go"), Outcome::Done);

        let tool_texts = |r: &(Vec<Message>, usize)| -> Vec<String> {
            r.0.iter().filter_map(|m| if let Message::Tool { content, .. } = m { Some(content.clone()) } else { None }).collect()
        };
        assert_eq!(tool_texts(&m.requests[0]), [big]); // the call that overflowed
        // the summary call after the overflow: the same history, with the result stubbed
        assert_eq!(m.requests[1].1, 0);
        assert_eq!(tool_texts(&m.requests[1]), ["[elided: 5KB bash output — events.jsonl seq 4]"]);
        assert_eq!(m.requests[1].0[0], Message::System("S".into()));
        assert_eq!(last_user(&m.requests[1]), format!("go\n\n{}", policy::summary_prompt(None)));
    }

    #[test]
    fn summary_failure_leaves_view() {
        // one read whose reply reports 2_600 of 3_000 tokens (≥ 85%) triggers stage 2; each way the summary can fail
        let interrupted = Ok(Reply { text: "partial".into(), interrupted: true, ..Default::default() });
        let cases = [
            ("sumnet", Err(LlmError::Network("reset".into())), "compaction failed: network error: reset"),
            ("sumint", interrupted, "compaction failed: interrupted"),
            ("sumempty", text("  "), "compaction failed: empty summary"),
            ("sumwrite", text("SUM"), "compaction failed: cannot write summary: "),
        ];
        for (tag, summary_reply, want) in cases {
            let mut m = Scripted::new(vec![read_f(1, 2_600), summary_reply, text("done")]);
            let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
            let mut a = agent(tag, Mode::Writes, false, &mut m, &mut ui, &cancel);
            a.budget = 3_000;
            write_files(&a.tools.root, 1, "x");
            if tag == "sumwrite" {
                fs::remove_dir_all(a.session.dir.join("summaries")).unwrap();
            }
            assert_eq!(a.run_request("go"), Outcome::Done, "{tag}: the loop goes on");
            let session = a.session;

            assert_eq!(m.requests.len(), 3, "{tag}");
            assert_eq!(m.requests[1].1, 0, "{tag}: the summary call");
            let e = errors(&session.events, "compaction");
            assert_eq!(e.len(), 1, "{tag}");
            assert!(e[0].0.starts_with(want) && !e[0].1, "{tag}: {e:?}");
            assert_eq!(ui.notices, ["compacting… (summarizing events 1–2)".to_string(), e[0].0.clone()], "{tag}");
            assert!(compactions(&session.events).is_empty(), "{tag}");
            // the view is unchanged: the next request is the history as it was, with no summary
            assert_eq!(m.requests[2].0[..2], m.requests[0].0[..], "{tag}");
            assert_eq!(m.requests[2].0.len(), 4, "{tag}: system, user, the read and its result");
            assert_eq!(ui.text, "done", "{tag}: nothing of the summary call is streamed");
        }
    }

    #[test]
    fn single_long_request_compacts_mid_turn() {
        // 30 reads of distinct 3_000-byte files, budget 30_000. Reply k reports 1_000·k tokens: stage 1 at reply 17 (≥ 60%),
        // stage 2 at reply 18 (still ≥ 60% right after stage 1); the 19th reply is the summary, and every later one reports 5_000.
        let mut script: Vec<_> = (1..=18).map(|i| read_f(i, 1_000 * i as u64)).collect();
        script.push(text("SUM"));
        script.extend((19..=30).map(|i| read_f(i, 5_000)));
        script.push(text("done"));
        let mut m = Scripted::new(script);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("midturn", Mode::Writes, false, &mut m, &mut ui, &cancel);
        a.budget = 30_000;
        write_files(&a.tools.root, 30, &format!("{}\n", "y".repeat(99)).repeat(30));
        assert_eq!(a.run_request("read all thirty files"), Outcome::Done);
        let session = a.session;

        assert_eq!(m.requests.len(), 32);
        assert_eq!(stages(&session.events), [1, 2]);
        let c = compactions(&session.events)[1];
        let Kind::Compaction { from_seq: Some(1), to_seq: Some(to), summary: Some(summary), .. } = &c.kind else { panic!("{c:?}") };
        assert!(summary.ends_with("\n\n## Current request (verbatim)\nread all thirty files"), "{summary}");
        // the cut falls inside the request, just before an assistant event
        assert!(matches!(session.events[*to as usize].kind, Kind::Assistant { .. }));
        assert_eq!(m.requests[18].1, 0); // the summary call
        let next = &m.requests[19].0;
        assert_eq!(next[1], Message::User(summary.clone()));
        assert!(matches!(next[2], Message::Assistant { .. }), "{:?}", next[2]);
    }

    #[test]
    fn nothing_to_compact_is_a_notice() {
        let mut m = Scripted::new(vec![]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("nothing", Mode::Writes, false, &mut m, &mut ui, &cancel);
        assert_eq!(a.compact(None), Ok(()));
        let session = a.session;

        assert_eq!(ui.notices, ["nothing to compact"]);
        assert_eq!(session.events.len(), 1);
        assert!(ui.events.is_empty() && m.requests.is_empty());
    }

    /// A text reply that reports `prompt` tokens.
    fn text_at(s: &str, prompt: u64) -> Result<Reply, LlmError> {
        Ok(Reply { usage: Usage { prompt, ..Usage::default() }, ..text(s).unwrap() })
    }

    #[test]
    fn text_only_requests_compact_before_the_next_call() {
        // budget 1_000; no tool calls anywhere. The second answer reports 900 tokens (≥ 85%), so the third request
        // compacts before its first model call.
        let mut m = Scripted::new(vec![text_at("a1", 100), text_at("a2", 900), text("SUM"), text("a3")]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("textonly", Mode::Writes, false, &mut m, &mut ui, &cancel);
        a.budget = 1_000;
        for request in ["first", "second", "third"] {
            assert_eq!(a.run_request(request), Outcome::Done, "{request}");
        }
        let session = a.session;

        assert_eq!(m.requests.len(), 4);
        assert_eq!(m.requests.iter().map(|r| r.1).collect::<Vec<_>>(), [tools::schemas().len(), tools::schemas().len(), 0, tools::schemas().len()]);
        assert_eq!(stages(&session.events), [2]);
        // the third request's call is sent from the summary of the first turn (cut A keeps the last two user turns)
        let third = &m.requests[3].0;
        assert!(
            matches!(&third[..], [Message::System(_), Message::User(u), Message::Assistant { text, .. }, Message::User(last)]
                if u.starts_with("[Session summary — events 1–3") && u.ends_with("\n\nsecond") && text == "a2" && last == "third"),
            "{third:?}"
        );
    }

    #[test]
    fn failed_stage2_is_not_retried_in_the_same_request() {
        // budget 3_000; every reply of the first request reports ≥ 85%. The first summary call fails; the rest of that request
        // makes no further summary call. The next request may try again (its first call is checked too), and succeeds.
        let mut m = Scripted::new(vec![
            read_f(1, 2_600),
            Err(LlmError::Network("reset".into())),
            read_f(2, 2_700),
            read_f(3, 2_800),
            text_at("done", 2_900),
            text("SUM"),
            text("ok"),
        ]);
        let (mut ui, cancel) = (TestUi::default(), AtomicBool::new(false));
        let mut a = agent("latch", Mode::Writes, false, &mut m, &mut ui, &cancel);
        a.budget = 3_000;
        write_files(&a.tools.root, 3, "x");
        assert_eq!(a.run_request("go"), Outcome::Done);
        let Agent { session, perms, tools, .. } = a;

        let summary_calls = |m: &Scripted| m.requests.iter().filter(|r| r.1 == 0).count();
        assert_eq!(m.requests.len(), 5); // read, failed summary, read, read, answer
        assert_eq!(summary_calls(&m), 1);
        assert_eq!(errors(&session.events, "compaction").len(), 1);
        // the summary call is announced with its range before it starts (a mid-turn cut: events 1–2 of the request)
        assert_eq!(ui.notices, ["compacting… (summarizing events 1–2)", "compaction failed: network error: reset"]);
        assert!(stages(&session.events).is_empty());

        // the latch is per request
        let mut a = Agent { session, model: &mut m, ui: &mut ui, perms, tools, budget: 3_000, cancel: &cancel, api_key_env: "XOLOTL_TEST_KEY", deadline: None };
        assert_eq!(a.run_request("again"), Outcome::Done);
        let session = a.session;
        assert_eq!(m.requests.len(), 7);
        assert_eq!(summary_calls(&m), 2);
        assert_eq!(stages(&session.events), [2]);
    }

    /// One real request against the API: `cargo test live_hello -- --ignored --nocapture`. The key comes from where the
    /// binary reads it (the env var, else the Windows User scope); without one the test is skipped.
    #[test]
    #[ignore]
    fn live_hello() {
        use crate::llm::{ZaiClient, ZaiConfig};
        use crate::ui::Headless;
        let cfg = crate::Config::default();
        let Some(api_key) = crate::api_key(&cfg.api_key_env) else {
            eprintln!("live_hello skipped: {} is not set", cfg.api_key_env);
            return;
        };
        let root = test_dir("live");
        let cwd = root.join("proj");
        fs::create_dir_all(&cwd).unwrap();
        let ctx = crate::Ctx { root: root.join("home"), cwd, shell: tools::resolve_shell(), model: cfg.model.clone(), budget: cfg.context_budget, allow: vec![], api_key_env: cfg.api_key_env.clone() };
        let session = crate::new_session(&ctx, Mode::Yolo).unwrap();
        let (perms, tools) = (Perms::new(Mode::Yolo, true, session.dir.clone(), vec![]), crate::tool_ctx(&ctx, &session));
        let mut model = ZaiClient::new(ZaiConfig {
            base_url: cfg.base_url,
            model: cfg.model,
            api_key,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            reasoning_effort: cfg.reasoning_effort,
        });
        let (mut ui, cancel) = (Headless { out: Vec::new() }, AtomicBool::new(false));
        let mut a = Agent { session, model: &mut model, ui: &mut ui, perms, tools, budget: ctx.budget, cancel: &cancel, api_key_env: &ctx.api_key_env, deadline: None };
        let outcome = a.run_request("Create a file named hello.txt containing exactly: hi");
        eprintln!("session: {}", a.session.dir.display());
        assert_eq!(outcome, Outcome::Done, "{}", String::from_utf8_lossy(&ui.out));
        assert_eq!(fs::read_to_string(ctx.cwd.join("hello.txt")).unwrap().trim(), "hi");
    }
}

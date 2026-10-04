//! Everything the model sees: system prompt, environment snapshot, output caps, and the message view built from
//! the on-disk events. Pure functions only (no I/O). Signatures follow spec §5.1 so a v2 optimizer can rewrite this file.

use crate::llm::{Message, ToolCall};
use crate::store::{Event, Kind};
use crate::tools::{Shell, clip, clip_tail};
use serde_json::Value;
use std::path::Path;

pub struct Limits {
    pub step_limit: usize,
    pub repeat_threshold: usize,
    pub keep_recent_results: usize,
    pub stage1_pct: usize,
    pub stage2_pct: usize,
    pub cut_keep_pct: usize,
    pub cut_mid_pct: usize,
    pub tool_cap: usize,
    pub bash_head: usize,
    pub bash_tail: usize,
    pub snapshot_timeout_s: u64,
    pub agents_md_max: usize,
    pub toc_user_chars: usize,
}

pub const LIMITS: Limits = Limits {
    step_limit: 100,
    repeat_threshold: 3,
    keep_recent_results: 10,
    stage1_pct: 60,
    stage2_pct: 85,
    cut_keep_pct: 40,
    cut_mid_pct: 20,
    tool_cap: 30_000,
    bash_head: 5_000,
    bash_tail: 25_000,
    snapshot_timeout_s: 15,
    agents_md_max: 20_000,
    toc_user_chars: 500,
};

/// Resend each assistant turn's `reasoning_content` verbatim (docs/glm-api-notes.md, decision 2: without it the model loses its own thinking).
pub const RESEND_REASONING: bool = true;

pub struct Capped {
    pub text: String,
    pub truncated: bool,
}

/// One compound command, POSIX-compatible (`$BASH_VERSION` is simply empty under `sh`).
const POSIX_SNAPSHOT: &str = r#"echo "os: $(uname -srm 2>/dev/null)"; echo "shell: bash $BASH_VERSION"; echo "cwd: $(pwd)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git branch: $(git branch --show-current)"; echo "git status:"; git status --short | head -20
  echo "recent commits:"; git log --oneline -5
else echo "git: not a repository"; fi
echo "top-level:"; ls -A | head -30
echo "manifests and implied commands:"
[ -f Cargo.toml ] && echo "  Cargo.toml → cargo build / cargo test"
[ -f package.json ] && { echo "  package.json scripts:"; grep -A15 '"scripts"' package.json | head -15; }
[ -f pyproject.toml ] && echo "  pyproject.toml → pytest"
[ -f go.mod ] && echo "  go.mod → go build ./... / go test ./..."
[ -f Makefile ] && { echo "  Makefile targets:"; grep -E '^[A-Za-z0-9_-]+:' Makefile | head -10; }
echo "toolchains:"
for t in rustc cargo node python3 python py java gcc; do command -v $t >/dev/null 2>&1 && echo "  $t: $($t --version 2>&1 | head -1)"; done
command -v go >/dev/null 2>&1 && echo "  go: $(go version)"
echo "memory: $(awk '/MemTotal/{print int($2/1024)" MB"}' /proc/meminfo 2>/dev/null || sysctl -n hw.memsize 2>/dev/null)"
"#;

const POWERSHELL_SNAPSHOT: &str = r#"$ErrorActionPreference = 'SilentlyContinue'
"os: $([System.Runtime.InteropServices.RuntimeInformation]::OSDescription)"; "shell: powershell $($PSVersionTable.PSVersion)"; "cwd: $(Get-Location)"
git rev-parse --is-inside-work-tree 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  "git branch: $(git branch --show-current)"; "git status:"; git status --short | Select-Object -First 20
  "recent commits:"; git log --oneline -5
} else { "git: not a repository" }
"top-level:"; Get-ChildItem -Force | Select-Object -First 30 -ExpandProperty Name
"manifests and implied commands:"
if (Test-Path Cargo.toml) { "  Cargo.toml -> cargo build / cargo test" }
if (Test-Path package.json) { "  package.json scripts:"; Select-String -Path package.json -Pattern '"scripts"' -Context 0,15 | Select-Object -First 1 | ForEach-Object { $_.Line; $_.Context.PostContext } }
if (Test-Path pyproject.toml) { "  pyproject.toml -> pytest" }
if (Test-Path go.mod) { "  go.mod -> go build ./... / go test ./..." }
if (Test-Path Makefile) { "  Makefile targets:"; Select-String -Path Makefile -Pattern '^[A-Za-z0-9_-]+:' | Select-Object -First 10 | ForEach-Object { $_.Line } }
"toolchains:"
foreach ($t in 'rustc','cargo','node','python3','python','py','java','gcc') { if (Get-Command $t) { "  ${t}: $(& $t --version 2>&1 | Select-Object -First 1)" } }
if (Get-Command go) { "  go: $(go version)" }
"memory: $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)) MB"
"#;

pub fn env_snapshot_command(shell: &Shell) -> String {
    match shell {
        Shell::Bash(_) | Shell::Sh => POSIX_SNAPSHOT,
        Shell::PowerShell => POWERSHELL_SNAPSHOT,
    }
    .to_string()
}

pub fn system_prompt(env: &str, agents_md: Option<&str>, session_dir: &Path) -> String {
    let mut p = format!(
        r#"You are Xolotl, a coding agent working in the user's project. You act through tools; the user sees your replies and a one-line log of each tool call.

Rules:
- Read a file before editing it. Prefer `edit` over `write` for existing files; keep edits small and exact.
- For tasks with more than two steps, keep a plan with the `todo` tool and update it as you go.
- After changing code, verify it: run the build, the tests, or the program with `bash`. Report what you ran and the result.
- `bash` runs a fresh process in the project root on every call (cd does not persist; chain with &&). It has no stdin and a timeout. Never start servers, watchers or other long-running processes.
- Find files with `glob`/`grep` instead of guessing paths.
- If a tool result says it was truncated or elided, the full text is on disk: `read`/`grep` the path it gives.
- Your full session history is at {dir}/events.jsonl (one JSON event per line; line number = seq). After a summary, use it to recover exact details.
- Be concise. When done, say what changed and how you verified it.

# Environment
{env}"#,
        dir = session_dir.display()
    );
    if let Some(md) = agents_md {
        p.push_str("\n# Project instructions (AGENTS.md)\n");
        p.push_str(clip(md, LIMITS.agents_md_max));
    }
    p
}

pub fn completion_check(original_request: &str) -> String {
    format!(
        r#"Before you finish, check your work against the original request:

<request>
{original_request}
</request>

1. Did you build and/or run the tests, and did they pass? If not, do it now.
2. Is any debug code, temp file or stray change left over? Remove it.
3. Then reply with a short summary of what changed and how you verified it."#
    )
}

/// What the model sees of a tool result. `bash` keeps head and tail (errors and summaries sit at the end); everything else keeps the head.
pub fn cap_output(tool: &str, output: &str) -> Capped {
    let (head, tail) = if tool == "bash" { (LIMITS.bash_head, LIMITS.bash_tail) } else { (LIMITS.tool_cap, 0) };
    if output.len() <= head + tail {
        return Capped { text: output.to_string(), truncated: false };
    }
    let text = if tail == 0 { clip(output, head).to_string() } else { format!("{}\n…[middle elided]…\n{}", clip(output, head), clip_tail(output, tail)) };
    Capped { text, truncated: true }
}

/// Messages for every model call: system prompt, the latest stage-2 summary (if any), then the events after that
/// summary's range, with stage-1 stubs applied.
pub fn build_view(events: &[Event]) -> Vec<Message> {
    view(events, None)
}

/// `build_view` with every tool result (except `todo`) stubbed: what the summary call sends after a context overflow,
/// since the full view is what overflowed.
pub fn stubbed_view(events: &[Event]) -> Vec<Message> {
    view(events, Some(events.last().map_or(0, |e| e.seq)))
}

/// `stub_upto`: `None` applies the stage-1 cutoff in force.
fn view(events: &[Event], stub_upto: Option<u64>) -> Vec<Message> {
    let mut view = Vec::new();
    if let Some(system) = events.iter().find_map(|e| if let Kind::SessionStart { system, .. } = &e.kind { Some(system) } else { None }) {
        view.push(Message::System(system.clone()));
    }
    let after = latest_stage2(events).map_or(0, |(to, summary)| {
        view.push(Message::User(summary.to_string()));
        to
    });
    view.extend(render(events, after, stub_upto.unwrap_or_else(|| stage1_upto(events, after))));
    merge_users(view) // a cut before a user event leaves the summary next to a User: one turn, not two
}

/// `(to_seq, summary)` of the latest stage-2 compaction.
fn latest_stage2(events: &[Event]) -> Option<(u64, &str)> {
    events.iter().rev().find_map(|e| match &e.kind {
        Kind::Compaction { stage: 2, to_seq: Some(to), summary: Some(summary), .. } => Some((*to, summary.as_str())),
        _ => None,
    })
}

/// The stage-1 cutoff in force for a view that starts after `after_seq`: the latest one, unless the summary already covers it (then 0).
fn stage1_upto(events: &[Event], after_seq: u64) -> u64 {
    let latest = events.iter().rev().find_map(|e| match &e.kind {
        Kind::Compaction { stage: 1, upto_seq: Some(upto), .. } => Some(*upto),
        _ => None,
    });
    latest.filter(|&u| u > after_seq).unwrap_or(0)
}

/// Events with `seq > after_seq` as messages; tool results with `seq <= stub_upto` (except `todo`) become stubs.
/// The result always satisfies the API's pairing rule: every assistant tool call has exactly one result right after it.
pub(crate) fn render(events: &[Event], after_seq: u64, stub_upto: u64) -> Vec<Message> {
    let msgs = events.iter().filter(|e| e.seq > after_seq).filter_map(|e| to_message(e, stub_upto)).collect();
    merge_users(pair_tools(msgs))
}

fn to_message(e: &Event, stub_upto: u64) -> Option<Message> {
    Some(match &e.kind {
        Kind::User { text } | Kind::Nudge { text, .. } => Message::User(text.clone()),
        Kind::Assistant { text, reasoning, tool_calls, .. } => Message::Assistant {
            text: text.clone(),
            reasoning: (RESEND_REASONING && !reasoning.is_empty()).then(|| reasoning.clone()),
            // arguments that are not JSON are sent as `{}` (the call's result already says why it failed); the event keeps them
            tool_calls: tool_calls
                .iter()
                .map(|c| if serde_json::from_str::<Value>(&c.args).is_ok() { c.clone() } else { ToolCall { args: "{}".into(), ..c.clone() } })
                .collect(),
        },
        Kind::ToolResult { call_id, name, output, artifact, bytes, .. } => Message::Tool {
            call_id: call_id.clone(),
            content: if e.seq <= stub_upto && name != "todo" { stub(e.seq, name, *bytes, artifact.as_deref()) } else { output.clone() },
        },
        _ => return None,
    })
}

fn stub(seq: u64, name: &str, bytes: u64, artifact: Option<&str>) -> String {
    let full = artifact.map_or(String::new(), |a| format!("; full: {a}"));
    format!("[elided: {}KB {name} output — events.jsonl seq {seq}{full}]", bytes.div_ceil(1024))
}

/// Keeps only tool results that answer the immediately preceding assistant message (once each), and adds a
/// synthetic result, right after the real ones, for every call that has none.
fn pair_tools(msgs: Vec<Message>) -> Vec<Message> {
    fn close(out: &mut Vec<Message>, pending: &mut Vec<String>) {
        out.extend(pending.drain(..).map(|call_id| Message::Tool { call_id, content: "cancelled (no result recorded)".into() }));
    }
    let mut out = Vec::with_capacity(msgs.len());
    let mut pending: Vec<String> = Vec::new(); // calls of the latest assistant message still without a result
    for m in msgs {
        match m {
            Message::Tool { call_id, content } => {
                if let Some(i) = pending.iter().position(|p| *p == call_id) {
                    pending.remove(i);
                    out.push(Message::Tool { call_id, content });
                } // else: orphan or duplicate, dropped
            }
            other => {
                close(&mut out, &mut pending);
                if let Message::Assistant { tool_calls, .. } = &other {
                    pending = tool_calls.iter().map(|c| c.id.clone()).collect();
                }
                out.push(other);
            }
        }
    }
    close(&mut out, &mut pending);
    out
}

fn merge_users(msgs: Vec<Message>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(msgs.len());
    for m in msgs {
        match (out.last_mut(), m) {
            (Some(Message::User(prev)), Message::User(next)) => {
                prev.push_str("\n\n");
                prev.push_str(&next);
            }
            (_, m) => out.push(m),
        }
    }
    out
}

// ---- size, compaction triggers, cut and summary layout (spec §10.2–10.4) ----

#[derive(Debug, PartialEq)]
pub enum Compaction {
    /// Shrink tool results at or before `upto_seq` to stubs (no LLM call).
    Stage1 { upto_seq: u64 },
    Stage2,
}

/// Events `from_seq..=to_seq` are replaced by a summary. `mid_turn`: the cut falls inside a user turn, just before an assistant event.
#[derive(Debug, PartialEq)]
pub struct Cut {
    pub from_seq: u64,
    pub to_seq: u64,
    pub mid_turn: bool,
}

/// `chars / 3`; deliberately over-counts code (spec §10.2).
fn tokens_of(msgs: impl IntoIterator<Item = Message>) -> usize {
    msgs.into_iter().map(|m| m.chars()).sum::<usize>() / 3
}

/// The last response's `prompt + completion` tokens plus `chars / 3` of what came after it. A compaction after that
/// response makes its usage stale, so then it is `chars / 3` of the whole view. A response without usage (an
/// interrupted one reports none) does not count as the last response.
pub fn estimate_tokens(events: &[Event]) -> usize {
    let last_assistant = events.iter().enumerate().rev().find_map(|(i, e)| match &e.kind {
        Kind::Assistant { usage, .. } if usage.prompt > 0 => Some((i, usage)),
        _ => None,
    });
    let last_compaction = events.iter().rposition(|e| matches!(e.kind, Kind::Compaction { .. }));
    match last_assistant {
        // Per event, not `render`: its pairing would drop, as orphans, the results of that response's own tool calls.
        Some((i, usage)) if last_compaction.is_none_or(|c| c < i) => {
            (usage.prompt + usage.completion) as usize + tokens_of(events[i + 1..].iter().filter_map(|e| to_message(e, 0)))
        }
        _ => tokens_of(build_view(events)),
    }
}

pub fn compaction_needed(prompt_tokens: usize, budget: usize, events: &[Event]) -> Option<Compaction> {
    let pct = prompt_tokens * 100 / budget.max(1);
    // Stage 1 just ran and the very next response is still big: shrinking again will not help.
    let stage1_just_ran = events.iter().rposition(|e| matches!(e.kind, Kind::Compaction { .. })).is_some_and(|i| {
        matches!(events[i].kind, Kind::Compaction { stage: 1, .. }) && events[i + 1..].iter().filter(|e| matches!(e.kind, Kind::Assistant { .. })).count() == 1
    });
    if pct >= LIMITS.stage2_pct || (pct >= LIMITS.stage1_pct && stage1_just_ran) {
        return Some(Compaction::Stage2);
    }
    if pct < LIMITS.stage1_pct {
        return None;
    }
    let after = latest_stage2(events).map_or(0, |(to, _)| to);
    let results: Vec<u64> = events.iter().filter(|e| e.seq > after && matches!(e.kind, Kind::ToolResult { .. })).map(|e| e.seq).collect();
    // the newest `keep_recent_results` stay verbatim; the one before them is the cutoff
    Some(match results.len().checked_sub(LIMITS.keep_recent_results + 1) {
        Some(i) if results[i] > stage1_upto(events, after) => Compaction::Stage1 { upto_seq: results[i] },
        _ => Compaction::Stage2, // too few results to shrink, or shrinking would change nothing
    })
}

/// Where stage 2 cuts the history (spec §10.4 step 1); `None` when there is nothing worth compacting.
pub fn stage2_cut(events: &[Event], budget: usize) -> Option<Cut> {
    let after = latest_stage2(events).map_or(0, |(to, _)| to);
    let from_seq = after + 1;
    let stub = stage1_upto(events, after);
    // ponytail: re-renders the kept tail for every candidate: O(assistants after the last user × history). Upgrade path: prefix sums of per-event chars.
    let fits = |c: u64, pct: usize| tokens_of(render(events, c, stub)) * 100 <= budget * pct;
    // a cut only counts if it swallows a user or assistant event (else the "summary" would cover nothing)
    let valid = |c: u64| events.iter().any(|e| (from_seq..=c).contains(&e.seq) && matches!(e.kind, Kind::User { .. } | Kind::Assistant { .. }));
    let cut = |to_seq, mid_turn| Cut { from_seq, to_seq, mid_turn };

    let users: Vec<u64> = events.iter().filter(|e| matches!(e.kind, Kind::User { .. })).map(|e| e.seq).collect();
    let n = users.len();
    // A: just before the second-to-last user event, B: just before the last one
    let ab: Vec<u64> = [n.checked_sub(2), n.checked_sub(1)].into_iter().flatten().map(|i| users[i].saturating_sub(1)).collect();
    if let Some(&c) = ab.iter().find(|&&c| valid(c) && fits(c, LIMITS.cut_keep_pct)) {
        return Some(cut(c, false));
    }
    // C: mid-turn, just before an assistant event after the last user: the earliest that fits, else the last
    let last_user = users.last().copied().unwrap_or(0);
    let before_assistant: Vec<u64> = events
        .iter()
        .filter(|e| e.seq > last_user && matches!(e.kind, Kind::Assistant { .. }))
        .map(|e| e.seq.saturating_sub(1))
        .filter(|&c| valid(c))
        .collect();
    match before_assistant.iter().copied().find(|&c| fits(c, LIMITS.cut_mid_pct)).or(before_assistant.last().copied()) {
        Some(c) => Some(cut(c, true)),
        // no reply after the last user event (a big paste): the last valid A/B cut, though over 40%, still compacts the rest
        None => ab.iter().rev().copied().find(|&c| valid(c)).map(|c| cut(c, false)),
    }
}

pub fn summary_prompt(focus: Option<&str>) -> String {
    let mut p = String::from(
        "Summarize this session so work can continue from your summary alone. The raw history stays on disk, so write what someone continuing the work needs. Use exactly these sections:
## Goal
## Current state
## Decisions and why
## Files touched
## Failed attempts and why they failed
## Open TODOs and next step
Be specific: file paths, function names, commands, exact error messages. Do not write a table of contents; Xolotl appends one. Do not call tools.",
    );
    if let Some(f) = focus {
        p.push_str("\nFocus: ");
        p.push_str(f);
    }
    p
}

/// Built by the harness from the events it is given, so the pointers into `events.jsonl` can't be hallucinated.
pub fn table_of_contents(events: &[Event], session_dir: &Path) -> String {
    let list = |lines: Vec<String>| if lines.is_empty() { "(none)".to_string() } else { lines.join("\n") };
    let users: Vec<String> = events
        .iter()
        .filter_map(|e| match &e.kind {
            Kind::User { text } => Some(format!("- seq {}: {}", e.seq, clip(text, LIMITS.toc_user_chars))),
            _ => None,
        })
        .collect();
    let files: Vec<String> = changed_files(events)
        .into_iter()
        .map(|(path, seqs)| format!("- {path} — seq {}", seqs.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")))
        .collect();
    let saved: Vec<String> = events
        .iter()
        .filter_map(|e| match &e.kind {
            Kind::ToolResult { artifact: Some(a), .. } => Some(format!("- {a} — seq {}", e.seq)),
            _ => None,
        })
        .collect();
    let todo: Vec<String> = latest_todo(events).into_iter().collect();
    format!(
        "## Table of contents (generated by Xolotl)\nRaw history: {}/events.jsonl\n\n### User messages\n{}\n\n### Files changed\n{}\n\n### Saved outputs\n{}\n\n### Todo\n{}",
        session_dir.display(),
        list(users),
        list(files),
        list(saved),
        list(todo)
    )
}

/// The message that replaces events `cut.from_seq..=cut.to_seq` in the view. The table of contents covers everything up to
/// `to_seq` (not only this range), because only the latest summary is shown.
pub fn summary_message(summary: &str, cut: &Cut, events: &[Event], session_dir: &Path) -> String {
    let covered = &events[..events.partition_point(|e| e.seq <= cut.to_seq)];
    let mut msg = format!(
        "[Session summary — events {}–{} compacted. Raw history: {}/events.jsonl (line = seq); read/grep it for exact details.]\n\n{}\n\n{}",
        cut.from_seq,
        cut.to_seq,
        session_dir.display(),
        without_toc(summary),
        table_of_contents(covered, session_dir)
    );
    // a mid-turn cut would otherwise leave the model without its task
    if cut.mid_turn
        && let Some(request) = covered.iter().rev().find_map(|e| if let Kind::User { text } = &e.kind { Some(text) } else { None })
    {
        msg.push_str("\n\n## Current request (verbatim)\n");
        msg.push_str(request);
    }
    msg
}

/// `summary` up to its first line that starts a `## Table of contents` (any case): a model that copies the previous
/// summary's table would otherwise leave two, with stale seqs in one.
fn without_toc(summary: &str) -> &str {
    let mut at = 0;
    for line in summary.split_inclusive('\n') {
        if line.get(..20).is_some_and(|h| h.eq_ignore_ascii_case("## table of contents")) {
            return summary[..at].trim_end();
        }
        at += line.len();
    }
    summary
}

/// Files a successful `edit`/`write` changed, in order of first change, with the seq of every change.
fn changed_files(events: &[Event]) -> Vec<(String, Vec<u64>)> {
    let mut files: Vec<(String, Vec<u64>)> = Vec::new();
    for (i, e) in events.iter().enumerate() {
        let Kind::ToolResult { call_id, name, ok: true, .. } = &e.kind else { continue };
        if name != "edit" && name != "write" {
            continue;
        }
        let call = events[..i].iter().rev().find_map(|p| match &p.kind {
            Kind::Assistant { tool_calls, .. } => tool_calls.iter().find(|c| c.id == *call_id),
            _ => None,
        });
        let path = call.and_then(|c| serde_json::from_str::<Value>(&c.args).ok()).and_then(|a| a.get("path")?.as_str().map(String::from));
        let Some(path) = path else { continue };
        match files.iter_mut().find(|(p, _)| *p == path) {
            Some((_, seqs)) => seqs.push(e.seq),
            None => files.push((path, vec![e.seq])),
        }
    }
    files
}

/// The list as the latest valid `todo` call left it, rendered by the tool itself (a call it rejects changed nothing).
fn latest_todo(events: &[Event]) -> Option<String> {
    let calls = events.iter().rev().filter_map(|e| if let Kind::Assistant { tool_calls, .. } = &e.kind { Some(tool_calls) } else { None });
    calls.flat_map(|calls| calls.iter().rev()).filter(|c| c.name == "todo").find_map(|c| {
        let out = crate::tools::todo(&serde_json::from_str::<Value>(&c.args).ok()?);
        out.ok.then_some(out.text)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ToolCall, Usage};
    use std::path::PathBuf;

    fn ev(seq: u64, kind: Kind) -> Event {
        Event { seq, t_ms: 0, kind }
    }
    fn user(s: &str) -> Kind {
        Kind::User { text: s.into() }
    }
    fn asst(text: &str, reasoning: &str) -> Kind {
        Kind::Assistant { text: text.into(), reasoning: reasoning.into(), tool_calls: vec![], usage: Usage::default(), finish_reason: "stop".into(), latency_ms: 0 }
    }
    fn asst_calls(calls: &[(&str, &str, &str)]) -> Kind {
        let tool_calls = calls.iter().map(|(id, name, args)| ToolCall { id: (*id).into(), name: (*name).into(), args: (*args).into() }).collect();
        Kind::Assistant { text: String::new(), reasoning: String::new(), tool_calls, usage: Usage::default(), finish_reason: "tool_calls".into(), latency_ms: 0 }
    }
    fn result(id: &str, name: &str, output: &str) -> Kind {
        Kind::ToolResult { call_id: id.into(), name: name.into(), ok: true, output: output.into(), artifact: None, bytes: output.len() as u64, duration_ms: 0, exit_code: None }
    }
    fn with_artifact(mut k: Kind, path: &str) -> Kind {
        if let Kind::ToolResult { artifact, .. } = &mut k {
            *artifact = Some(path.into());
        }
        k
    }
    fn session_start(system: &str) -> Kind {
        Kind::SessionStart { cwd: "/w".into(), model: "m".into(), mode: "writes".into(), policy_hash: "h".into(), system: system.into() }
    }
    fn compaction(stage: u8, upto_seq: Option<u64>, to_seq: Option<u64>, summary: Option<&str>) -> Kind {
        Kind::Compaction { stage, upto_seq, from_seq: Some(1), to_seq, summary_path: None, summary: summary.map(Into::into), tokens_before: None }
    }
    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall { id: id.into(), name: name.into(), args: "{}".into() }
    }
    fn tool(id: &str, content: &str) -> Message {
        Message::Tool { call_id: id.into(), content: content.into() }
    }
    fn asst_msg(text: &str, reasoning: Option<&str>, tool_calls: Vec<ToolCall>) -> Message {
        Message::Assistant { text: text.into(), reasoning: reasoning.map(Into::into), tool_calls }
    }
    /// Numbers the kinds 1, 2, 3, … (a seq is a 1-based line number).
    fn evs(kinds: Vec<Kind>) -> Vec<Event> {
        kinds.into_iter().enumerate().map(|(i, k)| ev(i as u64 + 1, k)).collect()
    }
    /// `n` assistant events without tool calls, `len` chars of text each.
    fn plain(n: usize, len: usize) -> Vec<Kind> {
        (0..n).map(|_| asst(&"a".repeat(len), "")).collect()
    }
    /// `n` × (assistant calling bash, its result of `len` bytes); call ids are `{prefix}0`, `{prefix}1`, …
    fn pairs(prefix: &str, n: usize, len: usize) -> Vec<Kind> {
        (0..n)
            .flat_map(|i| {
                let id = format!("{prefix}{i}");
                [asst_calls(&[(id.as_str(), "bash", "{}")]), result(&id, "bash", &"r".repeat(len))]
            })
            .collect()
    }
    fn with_usage(mut k: Kind, prompt: u64, completion: u64) -> Kind {
        if let Kind::Assistant { usage, .. } = &mut k {
            *usage = Usage { prompt, completion, cached: 0 };
        }
        k
    }
    fn failed(mut k: Kind) -> Kind {
        if let Kind::ToolResult { ok, .. } = &mut k {
            *ok = false;
        }
        k
    }
    fn result_seqs(events: &[Event]) -> Vec<u64> {
        events.iter().filter(|e| matches!(e.kind, Kind::ToolResult { .. })).map(|e| e.seq).collect()
    }

    #[test]
    fn system_prompt_parts() {
        let dir = Path::new("/tmp/sess");
        let p = system_prompt("os: testos", None, dir);
        assert!(p.contains("os: testos"));
        assert!(p.contains("/tmp/sess/events.jsonl"));
        assert!(!p.contains("# Project instructions (AGENTS.md)"));

        let p = system_prompt("os: testos", Some("use tabs"), dir);
        assert!(p.ends_with("\n# Project instructions (AGENTS.md)\nuse tabs"));

        let long = "x".repeat(25_000);
        let p = system_prompt("e", Some(&long), dir);
        let md = p.split("# Project instructions (AGENTS.md)\n").nth(1).unwrap();
        assert!(!md.is_empty() && md.len() <= 20_000);
    }

    #[test]
    fn snapshot_commands() {
        for shell in [Shell::Bash(PathBuf::from("bash")), Shell::Sh] {
            let c = env_snapshot_command(&shell);
            assert!(c.contains("git status --short") && c.contains("head -20"));
        }
        assert!(env_snapshot_command(&Shell::PowerShell).contains("Get-Location"));
    }

    #[test]
    fn cap_output_rules() {
        let short = cap_output("bash", "short");
        assert_eq!((short.text.as_str(), short.truncated), ("short", false));

        let s: String = (0..100_000).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let b = cap_output("bash", &s);
        assert_eq!(b.text, format!("{}\n…[middle elided]…\n{}", &s[..5_000], &s[75_000..]));
        assert!(b.truncated);

        let r = cap_output("read", &s);
        assert_eq!(r.text, s[..30_000]);
        assert!(r.truncated);

        // the cap is a limit, not a trigger: exactly 30_000 bytes passes through
        for tool in ["bash", "read"] {
            let exact = "a".repeat(30_000);
            assert!(!cap_output(tool, &exact).truncated);
            assert!(cap_output(tool, &format!("{exact}a")).truncated);
        }
    }

    #[test]
    fn cap_output_multibyte_boundary() {
        let s = "é".repeat(40_000); // 80_000 bytes, every char 2 bytes
        assert!(cap_output("bash", &s).truncated);
        assert!(cap_output("read", &s).truncated); // must not panic
        // a leading ASCII byte shifts every char to an odd offset, so the cut points land inside chars
        let s = format!("a{s}");
        assert!(cap_output("bash", &s).truncated);
        assert!(cap_output("read", &s).truncated);
    }

    #[test]
    fn view_basic_order_and_skips() {
        let events = vec![
            ev(1, session_start("S")),
            ev(2, user("a")),
            ev(3, asst_calls(&[("c1", "read", "{}")])),
            ev(4, Kind::Permission { call_id: "c1".into(), decision: "allow".into(), by: "rule".into() }),
            ev(5, result("c1", "read", "file text")),
            ev(6, Kind::Error { kind: "network".into(), message: "m".into(), retry: true }),
            ev(7, Kind::Nudge { reason: "empty".into(), text: "n".into() }),
            ev(8, asst("done", "")),
        ];
        assert_eq!(
            build_view(&events),
            vec![
                Message::System("S".into()),
                Message::User("a".into()),
                asst_msg("", None, vec![call("c1", "read")]),
                tool("c1", "file text"),
                Message::User("n".into()),
                asst_msg("done", None, vec![]),
            ]
        );
    }

    #[test]
    fn view_stage1_stubs() {
        let kb = "x".repeat(1_000);
        let events = vec![
            ev(1, session_start("S")),
            ev(2, user("go")),
            ev(3, asst_calls(&[("c1", "bash", "{}")])),
            ev(4, with_artifact(result("c1", "bash", &kb), "artifacts/c1.txt")),
            ev(5, asst_calls(&[("c2", "todo", "{}")])),
            ev(6, result("c2", "todo", "[ ] plan")),
            ev(7, asst_calls(&[("c3", "bash", "{}")])),
            ev(8, result("c3", "bash", &"y".repeat(1_500))),
            ev(9, compaction(1, Some(6), None, None)),
        ];
        let tool_contents = |msgs: Vec<Message>| -> Vec<String> {
            msgs.into_iter().filter_map(|m| if let Message::Tool { content, .. } = m { Some(content) } else { None }).collect()
        };
        assert_eq!(
            tool_contents(build_view(&events)),
            ["[elided: 1KB bash output — events.jsonl seq 4; full: artifacts/c1.txt]", "[ ] plan", &"y".repeat(1_500)]
        );
        // no artifact: the "; full: …" part is omitted; 1_500 bytes round up to 2KB; todo is never stubbed
        assert_eq!(
            tool_contents(render(&events, 0, 8)),
            ["[elided: 1KB bash output — events.jsonl seq 4; full: artifacts/c1.txt]", "[ ] plan", "[elided: 2KB bash output — events.jsonl seq 8]"]
        );
    }

    #[test]
    fn view_stage2_summary() {
        let events = vec![
            ev(1, session_start("S")),
            ev(2, user("u1")),
            ev(3, asst_calls(&[("c1", "bash", "{}")])),
            ev(4, result("c1", "bash", "old")),
            ev(5, asst("a1", "")),
            ev(6, user("u2")),
            ev(7, asst_calls(&[("c2", "bash", "{}")])),
            ev(8, result("c2", "bash", &"z".repeat(1_500))),
            ev(9, asst_calls(&[("c3", "bash", "{}")])),
            ev(10, result("c3", "bash", "new")),
            ev(11, compaction(2, None, Some(3), Some("OLD"))),
            ev(12, compaction(2, None, Some(5), Some("SUM"))), // the latest stage 2 wins
            ev(13, compaction(1, Some(4), None, None)),        // at or before the cut: no effect
            ev(14, compaction(1, Some(8), None, None)),        // after the cut: stubs seq 8
        ];
        assert_eq!(
            build_view(&events),
            vec![
                Message::System("S".into()),
                Message::User("SUM\n\nu2".into()), // cut before a user event: the summary and that user turn merge
                asst_msg("", None, vec![call("c2", "bash")]),
                tool("c2", "[elided: 2KB bash output — events.jsonl seq 8]"),
                asst_msg("", None, vec![call("c3", "bash")]),
                tool("c3", "new"),
            ]
        );
        // a stage-1 cutoff that is not past the cut is ignored (seq 8 stays verbatim)
        let view = build_view(&events[..13]);
        assert!(view.contains(&tool("c2", &"z".repeat(1_500))));

        // mid-turn cut (before an assistant event): the summary stays a separate User
        let mid = vec![
            ev(1, session_start("S")),
            ev(2, user("u1")),
            ev(3, asst_calls(&[("c1", "bash", "{}")])),
            ev(4, result("c1", "bash", "old")),
            ev(5, asst_calls(&[("c2", "bash", "{}")])),
            ev(6, result("c2", "bash", "new")),
            ev(7, compaction(2, None, Some(4), Some("SUM"))),
        ];
        assert_eq!(
            build_view(&mid),
            vec![
                Message::System("S".into()),
                Message::User("SUM".into()),
                asst_msg("", None, vec![call("c2", "bash")]),
                tool("c2", "new"),
            ]
        );
    }

    #[test]
    fn view_repairs_pairs_and_merges_users() {
        let events = vec![
            ev(1, session_start("S")),
            ev(2, user("a")),
            ev(3, Kind::Nudge { reason: "completion_check".into(), text: "b".into() }),
            ev(4, asst_calls(&[("c1", "read", "{}"), ("c2", "read", "{}")])),
            ev(5, result("c1", "read", "r1")),
            ev(6, result("c9", "read", "orphan")),
            ev(7, asst("done", "")),
        ];
        assert_eq!(
            build_view(&events),
            vec![
                Message::System("S".into()),
                Message::User("a\n\nb".into()),
                asst_msg("", None, vec![call("c1", "read"), call("c2", "read")]),
                tool("c1", "r1"),
                tool("c2", "cancelled (no result recorded)"),
                asst_msg("done", None, vec![]),
            ]
        );
        // a trailing assistant whose calls never got results is closed off too
        assert_eq!(
            render(&events[..4], 0, 0),
            vec![
                Message::User("a\n\nb".into()),
                asst_msg("", None, vec![call("c1", "read"), call("c2", "read")]),
                tool("c1", "cancelled (no result recorded)"),
                tool("c2", "cancelled (no result recorded)"),
            ]
        );
    }

    #[test]
    fn view_pairing_tricky_orderings() {
        let r = |events: Vec<Event>| render(&events, 0, 0);
        // a duplicate result for one call_id: the first is kept, the second dropped
        assert_eq!(
            r(vec![
                ev(1, asst_calls(&[("c1", "read", "{}")])),
                ev(2, result("c1", "read", "first")),
                ev(3, result("c1", "read", "dup")),
            ]),
            vec![asst_msg("", None, vec![call("c1", "read")]), tool("c1", "first")]
        );
        // a result arriving after an intervening nudge no longer answers the call: it is dropped, a synthetic one goes first
        assert_eq!(
            r(vec![
                ev(1, asst_calls(&[("c1", "read", "{}")])),
                ev(2, Kind::Nudge { reason: "empty".into(), text: "n".into() }),
                ev(3, result("c1", "read", "late")),
            ]),
            vec![asst_msg("", None, vec![call("c1", "read")]), tool("c1", "cancelled (no result recorded)"), Message::User("n".into())]
        );
        // two consecutive assistants with calls: the first is closed off, and a late result for it is dropped
        assert_eq!(
            r(vec![
                ev(1, asst_calls(&[("c1", "read", "{}")])),
                ev(2, asst_calls(&[("c2", "read", "{}")])),
                ev(3, result("c1", "read", "late")),
                ev(4, result("c2", "read", "r2")),
            ]),
            vec![
                asst_msg("", None, vec![call("c1", "read")]),
                tool("c1", "cancelled (no result recorded)"),
                asst_msg("", None, vec![call("c2", "read")]),
                tool("c2", "r2"),
            ]
        );
    }

    #[test]
    fn view_reasoning_follows_switch() {
        let events = vec![ev(1, asst("x", "r")), ev(2, asst("y", ""))];
        let want = if RESEND_REASONING { Some("r") } else { None };
        assert_eq!(render(&events, 0, 0), vec![asst_msg("x", want, vec![]), asst_msg("y", None, vec![])]);
    }

    // ---- size, triggers, cut, summary layout (Task 9) ----

    #[test]
    fn estimate_uses_usage_then_chars() {
        let head = || vec![session_start(&"s".repeat(60)), user(&"u".repeat(60))];
        let call = || with_usage(asst_calls(&[("c1", "bash", "{}")]), 1000, 200);
        let out = || result("c1", "bash", &"r".repeat(300));

        // the last response's usage, plus chars/3 of what came after it (the 300-byte result, even though its call is not in the tail)
        let events = evs([head(), vec![call(), out()]].concat());
        assert_eq!(estimate_tokens(&events), 1000 + 200 + 100);

        // a compaction after that response makes its usage stale: chars of the whole view / 3
        // (system 60 + user 60 + the call "bash" + "{}" = 6 + result 300 = 426)
        let events = evs([head(), vec![call(), compaction(1, Some(1), None, None), out()]].concat());
        assert_eq!(estimate_tokens(&events), 426 / 3);
        assert_eq!(estimate_tokens(&events), build_view(&events).iter().map(Message::chars).sum::<usize>() / 3);

        // a compaction before the response does not make its usage stale
        let events = evs([head(), vec![compaction(1, Some(1), None, None), call(), out()]].concat());
        assert_eq!(estimate_tokens(&events), 1000 + 200 + 100);

        // no response yet: the view
        assert_eq!(estimate_tokens(&evs(head())), 120 / 3);
    }

    #[test]
    fn estimate_skips_replies_without_usage() {
        let head = || vec![session_start(&"s".repeat(60)), user(&"u".repeat(60))];
        let call = || with_usage(asst_calls(&[("c1", "bash", "{}")]), 1000, 200);
        let out = || result("c1", "bash", &"r".repeat(300));
        let cut_off = "partial\n[interrupted]";

        // an interrupted reply reports zero usage: the response before it still counts, plus chars/3 of all that came after it
        let events = evs([head(), vec![call(), out(), asst(cut_off, "")]].concat());
        assert_eq!(estimate_tokens(&events), 1000 + 200 + (300 + cut_off.len()) / 3);

        // no response with usage at all: chars/3 of the view
        let events = evs([head(), vec![asst(cut_off, "")]].concat());
        assert_eq!(estimate_tokens(&events), tokens_of(build_view(&events)));
        assert!(estimate_tokens(&events) > 0);
    }

    #[test]
    fn triggers() {
        let base = |n: usize| evs([vec![session_start("S"), user("go")], pairs("a", n, 10)].concat()); // results at seq 4, 6, 8, …
        let twelve = base(12);
        let r = result_seqs(&twelve);
        assert_eq!(r[1], 6);

        assert_eq!(compaction_needed(500, 1000, &twelve), None);
        assert_eq!(compaction_needed(900, 1000, &twelve), Some(Compaction::Stage2));
        // 60% with 12 results: shrink everything up to the 11th-most-recent
        assert_eq!(compaction_needed(650, 1000, &twelve), Some(Compaction::Stage1 { upto_seq: 6 }));
        // exactly 11 results is enough (the oldest one); 10 is not
        assert_eq!(compaction_needed(650, 1000, &base(11)), Some(Compaction::Stage1 { upto_seq: 4 }));
        assert_eq!(compaction_needed(650, 1000, &base(10)), Some(Compaction::Stage2));
        assert_eq!(compaction_needed(650, 1000, &base(5)), Some(Compaction::Stage2));

        // the thresholds are inclusive: 60% and 85% trigger, one below does not
        assert_eq!(compaction_needed(599, 1000, &twelve), None);
        assert_eq!(compaction_needed(600, 1000, &twelve), Some(Compaction::Stage1 { upto_seq: 6 }));
        assert_eq!(compaction_needed(849, 1000, &twelve), Some(Compaction::Stage1 { upto_seq: 6 }));
        assert_eq!(compaction_needed(850, 1000, &twelve), Some(Compaction::Stage2));

        // only results after the latest stage-2 range count: 8 old ones (seq 4..=18 are covered by the summary), then n new ones
        let after_summary = |n: usize| {
            evs([vec![session_start("S"), user("go")], pairs("a", 8, 10), vec![compaction(2, None, Some(18), Some("SUM"))], pairs("b", n, 10)].concat())
        };
        assert_eq!(compaction_needed(650, 1000, &after_summary(5)), Some(Compaction::Stage2)); // 13 results in all, only 5 after seq 18
        assert_eq!(compaction_needed(650, 1000, &after_summary(12)), Some(Compaction::Stage1 { upto_seq: 23 })); // 2nd new result
    }

    #[test]
    fn stage1_then_still_high_goes_stage2() {
        let head = [vec![session_start("S"), user("go")], pairs("a", 12, 10)].concat(); // results at 4..=26; a fresh stage 1 would cut at seq 6

        // stage 1 (at an old, lower cutoff) just ran and the very next response is still ≥ 60%: Stage 2, although a larger cutoff exists
        let a = evs([head.clone(), vec![compaction(1, Some(4), None, None), asst("ok", "")]].concat());
        assert_eq!(compaction_needed(650, 1000, &a), Some(Compaction::Stage2));
        assert_eq!(compaction_needed(500, 1000, &a), None);
        // control: no response since that stage 1, so the ordinary rule applies
        let control = evs([head.clone(), vec![compaction(1, Some(4), None, None)]].concat());
        assert_eq!(compaction_needed(650, 1000, &control), Some(Compaction::Stage1 { upto_seq: 6 }));
        // the rule is about stage 1: right after a stage 2, a first response at 65% still gets the ordinary Stage 1
        let d = evs([head.clone(), vec![compaction(2, None, Some(2), Some("SUM")), asst("ok", "")]].concat());
        assert_eq!(compaction_needed(650, 1000, &d), Some(Compaction::Stage1 { upto_seq: 6 }));

        // …Compaction{stage:1}, Assistant, Assistant, then 11 more results: it is no longer the first response, and the new cutoff is larger
        let b = evs([head.clone(), vec![compaction(1, Some(6), None, None), asst("1", ""), asst("2", "")], pairs("b", 11, 10)].concat());
        let r = result_seqs(&b);
        let upto = r[r.len() - 11];
        assert!(upto > 6);
        assert_eq!(compaction_needed(650, 1000, &b), Some(Compaction::Stage1 { upto_seq: upto }));

        // a new cutoff that does not move past the current one would change nothing: Stage 2
        for current in [6, 8] {
            let c = evs([head.clone(), vec![compaction(1, Some(current), None, None), asst("1", ""), asst("2", "")]].concat());
            assert_eq!(compaction_needed(650, 1000, &c), Some(Compaction::Stage2), "current cutoff {current}");
        }
    }

    #[test]
    fn stage2_cut_prefers_two_user_turns() {
        // users at seq 2, 10, 20; everything is tiny against the budget
        let kinds = [vec![session_start("S"), user("u1")], plain(7, 10), vec![user("u2")], plain(9, 10), vec![user("u3")], plain(1, 10)].concat();
        let events = evs(kinds);
        assert_eq!(events[9].seq, 10);
        assert!(matches!(events[9].kind, Kind::User { .. }) && matches!(events[19].kind, Kind::User { .. }));
        assert_eq!(stage2_cut(&events, 100_000), Some(Cut { from_seq: 1, to_seq: 9, mid_turn: false }));

        // after an earlier stage-2 range ending at seq 5 the cut starts at 6
        let mut later = events.clone();
        later.push(ev(22, compaction(2, None, Some(5), Some("SUM"))));
        assert_eq!(stage2_cut(&later, 100_000), Some(Cut { from_seq: 6, to_seq: 9, mid_turn: false }));
    }

    #[test]
    fn stage2_cut_falls_back_to_last_user() {
        // users at 2, 10, 20. The turn at 10..=19 is big (9 × 400 chars), so keeping from seq 10 on would be over 40% of 1000 tokens (400);
        // keeping from seq 20 on (about 100 chars) is not.
        let kinds = [vec![session_start("S"), user("u1")], plain(7, 10), vec![user("u2")], plain(9, 400), vec![user("u3")], plain(1, 100)].concat();
        let events = evs(kinds);
        assert!(render(&events, 9, 0).iter().map(Message::chars).sum::<usize>() / 3 > 400);
        assert!(render(&events, 19, 0).iter().map(Message::chars).sum::<usize>() / 3 <= 400);
        assert_eq!(stage2_cut(&events, 1000), Some(Cut { from_seq: 1, to_seq: 19, mid_turn: false }));
    }

    #[test]
    fn stage2_cut_measures_the_tail_with_stage1_stubs() {
        // users at 2, 10, 20; the turn at 10..=19 holds three 3_000-byte results; 40% of 4_000 tokens is 4_800 chars
        let kinds = [vec![session_start("S"), user("u1")], plain(7, 10), vec![user("u2")], pairs("p", 3, 3_000), plain(3, 10), vec![user("u3")], plain(1, 10)].concat();
        let events = evs(kinds);
        assert_eq!(result_seqs(&events), [12, 14, 16]);
        // verbatim, keeping seq 10 on is too much, so the cut falls back to just before the last user
        assert_eq!(stage2_cut(&events, 4_000), Some(Cut { from_seq: 1, to_seq: 19, mid_turn: false }));
        // with those results stubbed (as the view will show them) it fits
        let mut stubbed = events.clone();
        stubbed.push(ev(22, compaction(1, Some(16), None, None)));
        assert_eq!(stage2_cut(&stubbed, 4_000), Some(Cut { from_seq: 1, to_seq: 9, mid_turn: false }));
    }

    #[test]
    fn stage2_cut_mid_turn_single_long_request() {
        // start, one user request, then 40 × (assistant call, 3_000-byte result): the assistants are at seq 3, 5, …, 81
        let events = evs([vec![session_start("S"), user("do everything")], pairs("c", 40, 3_000)].concat());
        let cut = stage2_cut(&events, 30_000).unwrap();
        assert!(cut.mid_turn);
        assert_eq!(cut.from_seq, 1);
        assert!(matches!(events[cut.to_seq as usize].kind, Kind::Assistant { .. })); // the event at to_seq + 1
        assert!(render(&events, cut.to_seq, 0).iter().map(Message::chars).sum::<usize>() / 3 <= 6_000); // 20% of the budget
        // the earliest qualifying assistant: each pair is 3_006 chars, 5 pairs ≈ 5_010 tokens fit in 6_000, 6 pairs ≈ 6_012 do not
        assert_eq!(cut, Cut { from_seq: 1, to_seq: 72, mid_turn: true });

        // nothing fits in 20% of the budget: the last assistant (seq 81)
        assert_eq!(stage2_cut(&events, 1_000), Some(Cut { from_seq: 1, to_seq: 80, mid_turn: true }));
    }

    #[test]
    fn stage2_cut_nothing_to_compact() {
        // B would be seq 1, which covers only SessionStart: not a valid cut
        assert_eq!(stage2_cut(&evs(vec![session_start("S"), user("hi")]), 100_000), None);
        // nothing left after an earlier summary that covers the whole history so far
        let covered = evs(vec![session_start("S"), user("hi"), asst("a", ""), compaction(2, None, Some(3), Some("SUM"))]);
        assert_eq!(stage2_cut(&covered, 100_000), None);
        // a cut only has to be valid on its own: A (seq 1) is not, B (seq 2) is
        let two_users = evs(vec![session_start("S"), user("a"), user("b"), asst("c", "")]);
        assert_eq!(stage2_cut(&two_users, 100_000), Some(Cut { from_seq: 1, to_seq: 2, mid_turn: false }));
    }

    #[test]
    fn stage2_cut_big_paste_falls_back_to_b() {
        // an earlier turn, then a pasted request far over 40% of the budget and no reply yet: B (seq 3) is valid but too big to
        // keep, and C has no assistant after the last user. B still compacts the earlier turn.
        let events = evs(vec![session_start("S"), user("u1"), asst("a1", ""), user(&"p".repeat(3_000))]);
        assert_eq!(stage2_cut(&events, 1_000), Some(Cut { from_seq: 1, to_seq: 3, mid_turn: false }));
        // still nothing when no A/B cut is valid either
        assert_eq!(stage2_cut(&evs(vec![session_start("S"), user(&"p".repeat(3_000))]), 1_000), None);
    }

    #[test]
    fn summary_drops_a_model_written_toc() {
        let events = evs(vec![session_start("S"), user("go"), asst("a", "")]);
        let cut = Cut { from_seq: 1, to_seq: 3, mid_turn: false };
        let dir = Path::new("/s");
        // the model copied the previous summary's table of contents (stale seqs) into its own
        let model = "## Goal\nfix it\n\n## table of contents (generated by Xolotl)\n- seq 99: stale\n\n### Todo\n(none)";
        let msg = summary_message(model, &cut, &events, dir);
        assert_eq!(msg.to_lowercase().matches("## table of contents").count(), 1, "{msg}");
        assert!(msg.contains("## Goal\nfix it\n\n## Table of contents (generated by Xolotl)") && !msg.contains("seq 99"), "{msg}");
        assert!(msg.ends_with(&table_of_contents(&events, dir)));
        assert!(summary_prompt(None).contains("Do not write a table of contents; Xolotl appends one."));
    }

    #[test]
    fn view_sends_invalid_args_as_an_empty_object() {
        let events = vec![
            ev(1, asst_calls(&[("c1", "read", "{bad"), ("c2", "read", r#"{"path":"a"}"#), ("c3", "read", "")])),
            ev(2, result("c1", "read", "invalid JSON arguments")),
            ev(3, result("c2", "read", "ok")),
            ev(4, result("c3", "read", "invalid JSON arguments")),
        ];
        let Message::Assistant { tool_calls, .. } = &render(&events, 0, 0)[0] else { panic!() };
        assert_eq!(tool_calls.iter().map(|c| c.args.as_str()).collect::<Vec<_>>(), ["{}", r#"{"path":"a"}"#, "{}"]);
    }

    #[test]
    fn stubbed_view_stubs_every_result() {
        let events = vec![
            ev(1, session_start("S")),
            ev(2, user("u1")),
            ev(3, asst_calls(&[("c1", "bash", "{}")])),
            ev(4, result("c1", "bash", "old")),
            ev(5, compaction(2, None, Some(4), Some("SUM"))),
            ev(6, user("u2")),
            ev(7, asst_calls(&[("c2", "bash", "{}"), ("c3", "todo", "{}")])),
            ev(8, result("c2", "bash", &"z".repeat(1_500))),
            ev(9, result("c3", "todo", "[ ] plan")),
        ];
        assert_eq!(
            stubbed_view(&events),
            vec![
                Message::System("S".into()),
                Message::User("SUM\n\nu2".into()),
                asst_msg("", None, vec![call("c2", "bash"), call("c3", "todo")]),
                tool("c2", "[elided: 2KB bash output — events.jsonl seq 8]"),
                tool("c3", "[ ] plan"),
            ]
        );
    }

    #[test]
    fn toc_and_summary_message() {
        let long = "a".repeat(600);
        let mut events = evs(vec![
            session_start("S"),                                                                       // 1
            user("fix parser"),                                                                       // 2
            asst_calls(&[("c1", "edit", r#"{"path":"src/a.rs","old":"x","new":"y"}"#)]),             // 3
            Kind::Permission { call_id: "c1".into(), decision: "allow".into(), by: "mode".into() },   // 4
            result("c1", "edit", "edited src/a.rs"),                                                  // 5
            asst_calls(&[("c2", "bash", r#"{"command":"cargo test"}"#)]),                             // 6
            with_artifact(result("c2", "bash", "long output"), "/s/artifacts/c3.txt"),                // 7
            asst_calls(&[("c3", "edit", r#"{"path":"src/a.rs","old":"y","new":"z"}"#)]),              // 8
            result("c3", "edit", "edited src/a.rs"),                                                  // 9
            asst_calls(&[("c4", "edit", r#"{"path":"src/b.rs"}"#), ("c5", "read", r#"{"path":"src/c.rs"}"#)]), // 10
            failed(result("c4", "edit", "no match")),                                                 // 11: a failed edit changed nothing
            result("c5", "read", "file"),                                                             // 12: a read changes nothing
            asst_calls(&[("c6", "todo", r#"{"items":[{"text":"old","status":"done"}]}"#)]),          // 13
            result("c6", "todo", "[x] old"),                                                          // 14
            asst_calls(&[("c7", "todo", r#"{"items":[{"text":"x","status":"pending"},{"text":"y","status":"in_progress"},{"text":"z","status":"done"}]}"#)]), // 15
            result("c7", "todo", "[ ] x\n[>] y\n[x] z"),                                              // 16
            user(&long),                                                                              // 17
            asst_calls(&[("c8", "write", r#"{"path":"src/new.rs","content":"fn main(){}"}"#)]),       // 18
            result("c8", "write", "wrote src/new.rs"),                                                // 19
            asst("done", ""),                                                                         // 20
        ]);
        let dir = Path::new("/s");

        let toc = table_of_contents(&events, dir);
        assert!(toc.starts_with("## Table of contents (generated by Xolotl)\nRaw history: /s/events.jsonl\n"));
        assert!(toc.contains("- seq 2: fix parser"));
        assert!(toc.contains("- src/a.rs — seq 5, 9")); // one line per file, every change in order
        assert!(toc.contains("- src/new.rs — seq 19")); // `write` counts too
        assert!(!toc.contains("src/b.rs") && !toc.contains("src/c.rs")); // failed edit, read
        assert!(toc.contains("- /s/artifacts/c3.txt — seq 7"));
        assert!(toc.contains("[ ] x\n[>] y\n[x] z")); // the latest todo call wins
        assert!(!toc.contains("[x] old"));
        // a long user message is cut to 500 bytes
        assert!(toc.contains(&format!("- seq 17: {}\n", "a".repeat(500))));
        assert!(!toc.contains(&"a".repeat(501)));
        // the sections come in the documented order
        let at = |h: &str| toc.find(h).unwrap_or_else(|| panic!("missing {h}"));
        assert!(at("### User messages") < at("### Files changed") && at("### Files changed") < at("### Saved outputs") && at("### Saved outputs") < at("### Todo"));
        // empty sections say so
        let bare = table_of_contents(&evs(vec![session_start("S")]), dir);
        assert_eq!(bare.matches("(none)").count(), 4);

        // summary_message: only events up to the cut (seq 16: before the long request and the write)
        let header = "[Session summary — events 1–16 compacted. Raw history: /s/events.jsonl (line = seq); read/grep it for exact details.]";
        let cut = Cut { from_seq: 1, to_seq: 16, mid_turn: false };
        let msg = summary_message("SUM", &cut, &events, dir);
        assert_eq!(msg, format!("{header}\n\nSUM\n\n{}", table_of_contents(&events[..16], dir)));
        assert!(!msg.contains("src/new.rs") && !msg.contains("## Current request"));

        // mid-turn: the active request (the last user event at or before the cut) follows in full, not cut to 500
        events.push(ev(21, user("a later request")));
        let cut = Cut { from_seq: 1, to_seq: 19, mid_turn: true };
        let msg = summary_message("SUM", &cut, &events, dir);
        assert!(msg.starts_with("[Session summary — events 1–19 compacted."));
        assert!(msg.contains("SUM") && msg.contains(&table_of_contents(&events[..19], dir)));
        assert!(msg.ends_with(&format!("\n\n## Current request (verbatim)\n{long}")));
        assert!(!msg.contains("a later request"));
    }

    #[test]
    fn summary_prompt_sections_and_focus() {
        let p = summary_prompt(None);
        for s in ["## Goal", "## Current state", "## Decisions and why", "## Files touched", "## Failed attempts and why they failed", "## Open TODOs and next step"] {
            assert!(p.contains(s), "missing {s}");
        }
        assert!(p.ends_with("Do not call tools.") && !p.contains("Focus:"));
        let p = summary_prompt(Some("parser"));
        assert!(p.ends_with("Do not call tools.\nFocus: parser"));
    }
}

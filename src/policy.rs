//! Everything the model sees: system prompt, environment snapshot, output caps, and the message view built from
//! the on-disk events. Pure functions only (no I/O). Signatures follow spec §5.1 so a v2 optimizer can rewrite this file.

use crate::llm::Message;
use crate::store::{Event, Kind};
use crate::tools::{Shell, clip, clip_tail};
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
    let mut view = Vec::new();
    if let Some(system) = events.iter().find_map(|e| if let Kind::SessionStart { system, .. } = &e.kind { Some(system) } else { None }) {
        view.push(Message::System(system.clone()));
    }
    let stage2 = events.iter().rev().find_map(|e| match &e.kind {
        Kind::Compaction { stage: 2, to_seq: Some(to), summary: Some(summary), .. } => Some((*to, summary)),
        _ => None,
    });
    let after = stage2.map_or(0, |(to, summary)| {
        view.push(Message::User(summary.clone()));
        to
    });
    let stage1 = events.iter().rev().find_map(|e| match &e.kind {
        Kind::Compaction { stage: 1, upto_seq: Some(upto), .. } => Some(*upto),
        _ => None,
    });
    view.extend(render(events, after, stage1.filter(|&u| u > after).unwrap_or(0)));
    merge_users(view) // a cut before a user event leaves the summary next to a User: one turn, not two
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
            tool_calls: tool_calls.clone(),
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
}

//! The `Ui` trait (spec §6) and its two renderers: `Headless` (one JSON line per event) and `Terminal` (inline,
//! into normal scrollback: streamed text, one line per tool call, single-keypress prompts, a status line).

use crate::llm::ToolCall;
use crate::store::Event;
use crate::tools::{ToolOut, clip};
use crossterm::event::{self, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::queue;
use crossterm::style::{ContentStyle, Print, PrintStyledContent, StyledContent, Stylize};
use crossterm::terminal::{self, Clear, ClearType};
use serde_json::Value;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Longest first-line summary of a tool call, in bytes.
const SUMMARY_MAX: usize = 60;
/// Longest diff or output line shown, in bytes.
const LINE_MAX: usize = 160;
const DIFF_LINES: usize = 20;
const BASH_TAIL_LINES: usize = 5;

#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub tool: String,
    pub summary: String,
    pub detail: Option<String>,
    /// What "always" would grant (`Perms::grant_preview`).
    pub always: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    Allow,
    Deny,
    Always,
}

#[derive(Debug, Clone, Default)]
pub struct Status {
    pub ctx_tokens: usize,
    pub budget: usize,
    pub tool_calls: usize,
    pub elapsed_ms: u64,
    pub cached_pct: u8,
}

pub trait Ui {
    fn text_delta(&mut self, s: &str);
    fn reasoning_delta(&mut self, s: &str);
    fn tool_start(&mut self, call: &ToolCall);
    fn tool_end(&mut self, call: &ToolCall, out: &ToolOut, duration_ms: u64);
    fn ask(&mut self, req: &PermissionRequest) -> Decision;
    fn confirm(&mut self, question: &str) -> bool;
    fn event(&mut self, e: &Event);
    fn notice(&mut self, msg: &str);
    fn status(&mut self, s: &Status);
}

// ---- pure helpers ----

fn plural(n: usize, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

/// `950`, `48.2k`, `200k`.
fn kfmt(n: usize) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let tenths = n.saturating_add(50) / 100;
    if tenths.is_multiple_of(10) { format!("{}k", tenths / 10) } else { format!("{}.{}k", tenths / 10, tenths % 10) }
}

/// First line of `s`, clipped to `SUMMARY_MAX` bytes.
fn first_line(s: &str) -> String {
    clip(s.trim().lines().next().unwrap_or(""), SUMMARY_MAX).to_string()
}

/// `l`, or its first `LINE_MAX` bytes plus `…`.
fn cut(l: &str) -> String {
    if l.len() > LINE_MAX { format!("{}…", clip(l, LINE_MAX)) } else { l.to_string() }
}

/// End of a CSI sequence whose parameters start at `cs[i]`: past the final byte, or at the first byte that
/// cannot belong to it.
fn skip_csi(cs: &[char], mut i: usize) -> usize {
    while let Some(&c) = cs.get(i) {
        match c {
            ' '..='?' => i += 1,       // parameter and intermediate bytes
            '@'..='~' => return i + 1, // final byte
            _ => return i,
        }
    }
    i
}

/// End of an OSC string whose payload starts at `cs[i]`: past BEL or `ESC \`. An ESC that is not part of `ESC \`
/// is left for the caller, as it starts a new sequence.
fn skip_osc(cs: &[char], mut i: usize) -> usize {
    while let Some(&c) = cs.get(i) {
        match c {
            '\x07' => return i + 1,
            '\x1b' => return if cs.get(i + 1) == Some(&'\\') { i + 2 } else { i },
            _ => i += 1,
        }
    }
    i
}

/// Text that is safe to print: model and tool output must not be able to move the cursor, clear or recolor the
/// screen, or retitle the window, because a prompt has to show what will actually run. Strips ANSI escape
/// sequences (CSI, OSC, any other `ESC x`, and the 8-bit CSI), drops `\r`, and turns every other control
/// character except `\n` and `\t` into `·`. A sequence split across two streamed chunks loses its ESC, so the
/// rest of it prints as harmless text.
fn sanitize(s: &str) -> String {
    let cs: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        i += 1;
        match c {
            '\x1b' => match cs.get(i) {
                Some('[') => i = skip_csi(&cs, i + 1),
                Some(']') => i = skip_osc(&cs, i + 1),
                Some(_) => i += 1,
                None => {}
            },
            '\u{9b}' => i = skip_csi(&cs, i),
            '\r' => {}
            '\n' | '\t' => out.push(c),
            c if c.is_control() => out.push('·'),
            c => out.push(c),
        }
    }
    out
}

/// A short description of the call's arguments for the tool line and the permission prompt. Paths are never
/// cut: the end of a path is what a person approving an edit needs to see. Unparseable or unexpected args fall
/// back to the first line of the raw text.
pub fn tool_summary(call: &ToolCall) -> String {
    let a = serde_json::from_str::<Value>(&call.args).unwrap_or(Value::Null); // Null: every lookup misses, so the fallback applies
    let s = |k: &str| a.get(k).and_then(Value::as_str);
    let n = |k: &str| a.get(k).and_then(Value::as_u64);
    let found = match call.name.as_str() {
        "bash" => s("command").map(first_line),
        "read" => s("path").map(|p| match (n("offset"), n("limit")) {
            (None, None) => p.to_string(),
            (offset, Some(limit)) => {
                let first = offset.unwrap_or(1);
                format!("{p}:{first}-{}", first.saturating_add(limit.saturating_sub(1)))
            }
            (Some(first), None) => format!("{p}:{first}-"),
        }),
        "grep" => s("regex").map(|r| format!("\"{}\" {}", clip(r, SUMMARY_MAX), s("path").unwrap_or("."))),
        "glob" => s("pattern").map(|p| clip(p, SUMMARY_MAX).to_string()),
        "edit" | "write" => s("path").map(str::to_string),
        "todo" => a.get("items").and_then(Value::as_array).map(|i| plural(i.len(), "item")),
        _ => None,
    };
    // one line, however the model wrote the arguments (a lone `\r` would otherwise survive `first_line`)
    let one_line = |s: String| s.split(['\r', '\n']).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ");
    one_line(found.unwrap_or_else(|| first_line(&call.args)))
}

/// `- ` lines for `old`, then `+ ` lines for `new`; after `max_lines` lines, `… (N more)`.
pub fn mini_diff(old: &str, new: &str, max_lines: usize) -> String {
    let all: Vec<String> = old
        .lines()
        .map(|l| format!("- {}", cut(l)))
        .chain(new.lines().map(|l| format!("+ {}", cut(l))))
        .collect();
    let mut shown: Vec<String> = all.iter().take(max_lines).cloned().collect();
    if all.len() > max_lines {
        shown.push(format!("… ({} more)", all.len() - max_lines));
    }
    shown.join("\n")
}

pub fn status_line(s: &Status) -> String {
    format!(
        "── {} / {} ctx · {} · {}s · {}% cached ──",
        kfmt(s.ctx_tokens),
        kfmt(s.budget),
        plural(s.tool_calls, "tool"),
        s.elapsed_ms / 1000,
        s.cached_pct
    )
}

/// The permission prompt's last line; `[a]lways` names what it grants.
fn ask_prompt(req: &PermissionRequest) -> String {
    let always = if req.always.is_empty() { String::new() } else { format!(" ({})", req.always) };
    format!("    allow {}? [y]es [n]o [a]lways{always} › ", req.tool)
}

/// What a keypress at a permission prompt means: the decision, and whether it also cancels the run (Ctrl+C).
pub fn key_decision(code: KeyCode, mods: KeyModifiers) -> Option<(Decision, bool)> {
    let ctrl = mods.contains(KeyModifiers::CONTROL);
    match code {
        KeyCode::Esc => Some((Decision::Deny, false)),
        KeyCode::Char(c) => match c.to_ascii_lowercase() {
            'c' if ctrl => Some((Decision::Deny, true)),
            _ if ctrl || mods.contains(KeyModifiers::ALT) => None,
            'y' => Some((Decision::Allow, false)),
            'n' => Some((Decision::Deny, false)),
            'a' => Some((Decision::Always, false)),
            _ => None,
        },
        _ => None,
    }
}

/// The text appended to a tool line once the call finishes, and its style.
fn outcome(call: &ToolCall, out: &ToolOut, duration_ms: u64) -> (String, Sty) {
    let bash = call.name == "bash";
    if out.cancelled {
        return ("cancelled".into(), Sty::Yellow);
    }
    if let (true, Some(code)) = (bash, out.exit_code) {
        return (format!("exit {code} · {:.1}s", duration_ms as f64 / 1000.0), if code == 0 { Sty::Dim } else { Sty::Red });
    }
    if !out.ok {
        // bash without an exit code (timeout, refused): the reason is the footer on the last line
        let msg = if bash { out.text.lines().last() } else { out.text.lines().next() }.unwrap_or("");
        return (format!("✗ {}", first_line(msg.trim_matches(['[', ']']))), Sty::Red);
    }
    if matches!(call.name.as_str(), "grep" | "glob") {
        // ponytail: counts result lines, so grep context lines are counted too and a capped list reports the cap.
        let n = out.text.lines().filter(|l| !l.starts_with('(') && *l != "--").count();
        let msg = match n {
            0 => "no matches".to_string(),
            1 => "1 match".to_string(),
            n => format!("{n} matches"),
        };
        return (msg, Sty::Dim);
    }
    ("ok".into(), Sty::Dim)
}

// ---- Headless ----

/// Scripted use (`-p`): every event is one JSON line on `out`. Nothing is rendered and nothing is asked.
pub struct Headless<W: Write> {
    pub out: W,
}

impl<W: Write> Ui for Headless<W> {
    fn text_delta(&mut self, _: &str) {}
    fn reasoning_delta(&mut self, _: &str) {}
    fn tool_start(&mut self, _: &ToolCall) {}
    fn tool_end(&mut self, _: &ToolCall, _: &ToolOut, _: u64) {}
    fn ask(&mut self, _: &PermissionRequest) -> Decision {
        Decision::Deny
    }
    fn confirm(&mut self, _: &str) -> bool {
        false
    }
    fn event(&mut self, e: &Event) {
        // the session store keeps the authoritative copy, so a closed pipe must not abort the run
        if let Ok(line) = serde_json::to_string(e) {
            let _ = writeln!(self.out, "{line}");
            let _ = self.out.flush();
        }
    }
    fn notice(&mut self, msg: &str) {
        let _ = writeln!(io::stderr(), "{msg}"); // a closed stderr must not abort the run
    }
    fn status(&mut self, _: &Status) {}
}

// ---- Terminal ----

#[derive(Clone, Copy)]
enum Sty {
    Plain,
    Dim,
    Red,
    Green,
    Yellow,
    DimYellow,
    Cyan,
}

impl Sty {
    fn style(self) -> ContentStyle {
        let s = ContentStyle::new();
        match self {
            Sty::Plain => s,
            Sty::Dim => s.dim(),
            Sty::Red => s.red(),
            Sty::Green => s.green(),
            Sty::Yellow => s.yellow(),
            Sty::DimYellow => s.yellow().dim(),
            Sty::Cyan => s.cyan(),
        }
    }
}

/// What the cursor's current line holds.
#[derive(Clone, Copy, PartialEq)]
enum Line {
    /// At the start of a line.
    Bol,
    /// Streamed answer text or a prompt, mid-line.
    Text,
    /// Raw reasoning text (`show_thinking`), mid-line.
    Reasoning,
    /// The one-line `⋯ thinking` counter, redrawn in place; holds the char count.
    Thinking(usize),
    /// A tool line waiting for its outcome.
    Tool,
}

/// Raw mode for the duration of one prompt; dropped on every exit path, panics included.
struct RawMode;

impl RawMode {
    fn enter() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(RawMode)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

/// Blocks for one accepted keypress. `always` is false for prompts that do not offer `[a]lways`.
fn read_key(always: bool) -> io::Result<(Decision, bool)> {
    let _raw = RawMode::enter()?;
    // type-ahead from before the prompt appeared must not answer it
    while event::poll(Duration::ZERO)? {
        event::read()?;
    }
    loop {
        // Windows reports key release as well as press
        if let event::Event::Key(k) = event::read()?
            && k.kind == KeyEventKind::Press
            && let Some((d, cancel)) = key_decision(k.code, k.modifiers)
            && (always || d != Decision::Always)
        {
            return Ok((d, cancel));
        }
    }
}

/// Interactive use: renders inline into the normal scrollback. Colors and in-place redraws only when stdout is
/// a terminal.
pub struct Terminal {
    pub show_thinking: bool,
    pub cancel: &'static AtomicBool,
    tty: bool,
    line: Line,
}

impl Terminal {
    pub fn new(show_thinking: bool, cancel: &'static AtomicBool) -> Self {
        Terminal { show_thinking, cancel, tty: io::stdout().is_terminal(), line: Line::Bol }
    }

    /// Everything printed to the user goes through here, and through `sanitize`. Only this type's own escape
    /// codes (styles, the thinking-line redraw) reach the terminal unfiltered.
    fn put(&mut self, s: &str, sty: Sty) {
        let s = sanitize(s);
        let s = s.as_str();
        if s.is_empty() {
            return;
        }
        let mut o = io::stdout().lock();
        let _ = if self.tty { queue!(o, PrintStyledContent(StyledContent::new(sty.style(), s))) } else { queue!(o, Print(s)) };
        let _ = o.flush();
        self.line = if s.ends_with('\n') { Line::Bol } else { Line::Text };
    }

    /// Finishes whatever is on the current line, so the next output starts on a fresh one.
    fn end_line(&mut self) {
        match self.line {
            Line::Bol => {}
            Line::Thinking(n) if !self.tty => {
                self.draw_thinking(n);
                self.put("\n", Sty::Plain);
            }
            _ => self.put("\n", Sty::Plain),
        }
        self.line = Line::Bol;
    }

    fn draw_thinking(&mut self, chars: usize) {
        if self.tty {
            let _ = queue!(io::stdout(), Print("\r"), Clear(ClearType::CurrentLine));
        }
        self.put(&format!("  ⋯ thinking ({} chars)", kfmt(chars)), Sty::Dim);
        self.line = Line::Thinking(chars);
    }

    /// Indented lines; a diff is colored by its `+ ` / `- ` prefix, anything else is dim.
    fn block(&mut self, text: &str, diff: bool) {
        for l in text.lines() {
            let sty = match (diff, l.get(..2)) {
                (true, Some("+ ")) => Sty::Green,
                (true, Some("- ")) => Sty::Red,
                _ => Sty::Dim,
            };
            // diff lines arrive already cut by `mini_diff`; a prompt's detail is shown whole
            self.put(&format!("    {}\n", if diff { l.to_string() } else { cut(l) }), sty);
        }
    }

    /// Shows `text`, then waits for one keypress. No usable terminal means no answer, which is a denial.
    fn prompt(&mut self, text: &str, always: bool) -> Decision {
        self.put(text, Sty::Yellow);
        let (d, cancel) = read_key(always).unwrap_or((Decision::Deny, false));
        if cancel {
            self.cancel.store(true, Ordering::SeqCst);
        }
        let (label, sty) = match (d, cancel) {
            (_, true) => ("cancel", Sty::Yellow),
            (Decision::Allow, _) => ("yes", Sty::Green),
            (Decision::Deny, _) => ("no", Sty::Red),
            (Decision::Always, _) => ("always", Sty::Cyan),
        };
        self.put(label, sty);
        self.put("\n", Sty::Plain);
        d
    }
}

impl Ui for Terminal {
    fn text_delta(&mut self, s: &str) {
        if matches!(self.line, Line::Thinking(_) | Line::Reasoning | Line::Tool) {
            self.end_line();
        }
        self.put(s, Sty::Plain);
    }

    fn reasoning_delta(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        if self.show_thinking {
            if matches!(self.line, Line::Thinking(_) | Line::Tool) {
                self.end_line();
            }
            self.put(s, Sty::Dim);
            self.line = if s.ends_with('\n') { Line::Bol } else { Line::Reasoning };
            return;
        }
        let before = match self.line {
            Line::Thinking(n) => n,
            _ => {
                self.end_line();
                0
            }
        };
        let chars = before + s.chars().count();
        self.line = Line::Thinking(chars);
        if self.tty {
            self.draw_thinking(chars); // piped output gets the line once, from `end_line`
        }
    }

    fn tool_start(&mut self, call: &ToolCall) {
        self.end_line();
        self.put("  ● ", Sty::Cyan);
        self.put(&call.name, Sty::Plain);
        self.put(&format!("  {}", tool_summary(call)), Sty::Dim);
        self.line = Line::Tool;
    }

    fn tool_end(&mut self, call: &ToolCall, out: &ToolOut, duration_ms: u64) {
        let (text, sty) = outcome(call, out, duration_ms);
        if self.line == Line::Tool {
            self.put("  ", Sty::Plain);
        } else {
            // something was printed since `tool_start` (a prompt, a notice): the outcome gets its own line
            self.end_line();
            self.put("    ↳ ", Sty::Dim);
        }
        self.put(&text, sty);
        self.put("\n", Sty::Plain);

        let args = serde_json::from_str::<Value>(&call.args).unwrap_or(Value::Null);
        let arg = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("");
        match call.name.as_str() {
            "edit" if out.ok => self.block(&mini_diff(arg("old"), arg("new"), DIFF_LINES), true),
            "write" if out.ok => self.block(&mini_diff("", arg("content"), DIFF_LINES), true),
            "bash" => {
                // the text ends with a one-line `[exit N · Xs]` footer, which the outcome already shows
                let body = out.text.rsplit_once('\n').map_or("", |(body, _)| body).trim_end();
                let lines: Vec<&str> = body.lines().collect();
                let tail = lines.len().saturating_sub(BASH_TAIL_LINES);
                self.block(&lines[tail..].join("\n"), false);
            }
            _ => {}
        }
    }

    fn ask(&mut self, req: &PermissionRequest) -> Decision {
        self.end_line();
        self.put("  ? ", Sty::Yellow);
        self.put(&req.tool, Sty::Plain);
        self.put(&format!("  {}\n", req.summary), Sty::Dim);
        if let Some(detail) = &req.detail {
            self.block(detail, true);
        }
        self.prompt(&ask_prompt(req), true)
    }

    fn confirm(&mut self, question: &str) -> bool {
        self.end_line();
        self.prompt(&format!("    {question} [y]es [n]o › "), false) == Decision::Allow
    }

    fn event(&mut self, _: &Event) {}

    fn notice(&mut self, msg: &str) {
        self.end_line();
        self.put(&format!("{msg}\n"), Sty::DimYellow);
    }

    fn status(&mut self, s: &Status) {
        self.end_line();
        self.put(&format!("{}\n", status_line(s)), Sty::Dim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Kind;
    use serde_json::{Value, json};

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall { id: "c1".into(), name: name.into(), args: args.into() }
    }
    fn summary(name: &str, args: Value) -> String {
        tool_summary(&call(name, &args.to_string()))
    }

    #[test]
    fn tool_summaries() {
        assert_eq!(summary("bash", json!({ "command": "cargo test parser\nsecond line" })), "cargo test parser");
        assert_eq!(summary("bash", json!({ "command": "x".repeat(100) })), "x".repeat(60));
        // 100 two-byte chars: the cut lands on a char boundary
        assert_eq!(summary("bash", json!({ "command": "é".repeat(100) })).chars().count(), 30);

        assert_eq!(summary("read", json!({ "path": "src/p.rs", "offset": 80, "limit": 60 })), "src/p.rs:80-139");
        assert_eq!(summary("read", json!({ "path": "a" })), "a");
        assert_eq!(summary("read", json!({ "path": "a", "limit": 10 })), "a:1-10");
        assert_eq!(summary("read", json!({ "path": "a", "offset": 5 })), "a:5-");
        // model-supplied numbers never overflow
        summary("read", json!({ "path": "a", "offset": u64::MAX, "limit": u64::MAX }));

        assert_eq!(summary("grep", json!({ "regex": "fn x", "path": "src" })), "\"fn x\" src");
        assert_eq!(summary("grep", json!({ "regex": "fn x" })), "\"fn x\" .");
        assert_eq!(summary("glob", json!({ "pattern": "**/*.rs" })), "**/*.rs");
        assert_eq!(summary("edit", json!({ "path": "src/p.rs", "old": "a", "new": "b" })), "src/p.rs");
        assert_eq!(summary("write", json!({ "path": "src/q.rs", "content": "x" })), "src/q.rs");
        let item = json!({ "text": "t", "status": "pending" });
        assert_eq!(summary("todo", json!({ "items": [item, item, item] })), "3 items");
        assert_eq!(summary("todo", json!({ "items": [item] })), "1 item");

        // anything unparseable or unexpected falls back to the raw args
        assert_eq!(tool_summary(&call("read", "{bad")), "{bad");
        assert_eq!(tool_summary(&call("bash", "{}")), "{}");
        assert_eq!(summary("frob", json!({ "a": 1 })), r#"{"a":1}"#);

        // a tool line stays one line, whatever the model put in the arguments
        assert_eq!(summary("edit", json!({ "path": "a\nb" })), "a b");
        assert_eq!(summary("edit", json!({ "path": "a\r\nb\rc" })), "a b c");
        assert_eq!(summary("bash", json!({ "command": "ls\rrm -rf x" })), "ls rm -rf x");
        assert_eq!(tool_summary(&call("read", "{bad\rmore")), "{bad more");
    }

    #[test]
    fn sanitize_strips_terminal_control() {
        assert_eq!(sanitize("a\x1b[2K\rb"), "ab");
        assert_eq!(sanitize("x\x1b]0;title\x07y"), "xy");
        assert_eq!(sanitize("x\x1b]0;title\x1b\\y"), "xy");
        assert_eq!(sanitize("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(sanitize("a\x1b[1;38;5;196mb\x1b[?25lc"), "abc");
        assert_eq!(sanitize("a\u{9b}31mb"), "ab"); // 8-bit C1 CSI
        assert_eq!(sanitize("a\x1bcb"), "ab"); // two-byte ESC x
        assert_eq!(sanitize("a\x08b\x00c"), "a·b·c");
        assert_eq!(sanitize("a\u{85}b\x7fc"), "a·b·c");
        assert_eq!(sanitize("keep\n\ttabs é✓"), "keep\n\ttabs é✓");
        // unterminated sequences are dropped, and never panic
        assert_eq!(sanitize("a\x1b["), "a");
        assert_eq!(sanitize("a\x1b[31"), "a");
        assert_eq!(sanitize("a\x1b]0;never ends"), "a");
        assert_eq!(sanitize("a\x1b"), "a");
        // a malformed sequence stops at the first byte that cannot belong to it
        assert_eq!(sanitize("a\x1b[\x1b[31mb"), "ab");
        assert_eq!(sanitize("a\x1b[3\nb"), "a\nb");
        // an escape inside an OSC that is not ST starts a new sequence
        assert_eq!(sanitize("a\x1b]0;t\x1b[31mb"), "ab");
        // nothing that survives can be interpreted by a terminal
        let nasty: String = (0u32..0x200).filter_map(char::from_u32).collect();
        let clean = sanitize(&nasty);
        assert!(clean.chars().all(|c| !c.is_control() || c == '\n' || c == '\t'), "{clean:?}");
    }

    #[test]
    fn mini_diff_format() {
        assert_eq!(mini_diff("a\nb", "c", 20), "- a\n- b\n+ c");
        let old: Vec<String> = (1..=30).map(|i| format!("l{i}")).collect();
        assert_eq!(mini_diff(&old.join("\n"), "x", 5), "- l1\n- l2\n- l3\n- l4\n- l5\n… (26 more)");
        assert_eq!(mini_diff("", "", 5), "");
        // a very long line is cut on a char boundary, with a marker
        let cut = mini_diff(&"é".repeat(500), "", 5);
        assert!(cut.starts_with("- é") && cut.ends_with('…') && cut.len() < 200, "{cut}");
    }

    #[test]
    fn status_line_format() {
        let s = Status { ctx_tokens: 48_200, budget: 200_000, tool_calls: 7, elapsed_ms: 41_000, cached_pct: 62 };
        assert_eq!(status_line(&s), "── 48.2k / 200k ctx · 7 tools · 41s · 62% cached ──");
        let s = Status { ctx_tokens: 950, budget: 1_000, tool_calls: 1, elapsed_ms: 400, cached_pct: 0 };
        assert_eq!(status_line(&s), "── 950 / 1k ctx · 1 tool · 0s · 0% cached ──");
    }

    #[test]
    fn keys() {
        use Decision::*;
        let (none, ctrl) = (KeyModifiers::NONE, KeyModifiers::CONTROL);
        assert_eq!(key_decision(KeyCode::Char('y'), none), Some((Allow, false)));
        assert_eq!(key_decision(KeyCode::Char('n'), none), Some((Deny, false)));
        assert_eq!(key_decision(KeyCode::Char('a'), none), Some((Always, false)));
        assert_eq!(key_decision(KeyCode::Char('Y'), KeyModifiers::SHIFT), Some((Allow, false)));
        assert_eq!(key_decision(KeyCode::Esc, none), Some((Deny, false)));
        assert_eq!(key_decision(KeyCode::Char('c'), ctrl), Some((Deny, true)));
        assert_eq!(key_decision(KeyCode::Char('x'), none), None);
        assert_eq!(key_decision(KeyCode::Enter, none), None);
        assert_eq!(key_decision(KeyCode::Char('y'), ctrl), None); // Ctrl+Y is not "yes"
    }

    #[test]
    fn outcomes() {
        let ms = 3_100;
        let bash = call("bash", r#"{"command":"cargo test"}"#);
        let with = |exit_code, ok, text: &str, cancelled| ToolOut { ok, text: text.into(), exit_code, cancelled };
        assert_eq!(outcome(&bash, &with(Some(101), false, "x\n[exit 101 · 3.1s]", false), ms).0, "exit 101 · 3.1s");
        assert_eq!(outcome(&bash, &with(Some(0), true, "\n[exit 0 · 3.1s]", false), ms).0, "exit 0 · 3.1s");
        assert_eq!(outcome(&bash, &with(None, false, "x\n[timed out after 1s · partial output above]", false), ms).0, "✗ timed out after 1s · partial output above");
        assert_eq!(outcome(&bash, &with(None, false, "x\n[cancelled by user]", true), ms).0, "cancelled");
        let read = call("read", r#"{"path":"nope"}"#);
        assert_eq!(outcome(&read, &ToolOut::err("file not found: nope\nsecond"), ms).0, "✗ file not found: nope");
        assert_eq!(outcome(&read, &ToolOut::ok("1\ta"), ms).0, "ok");
        let grep = call("grep", r#"{"regex":"x"}"#);
        assert_eq!(outcome(&grep, &ToolOut::ok("a.rs\nb.rs\n(200 of 205 shown)"), ms).0, "2 matches");
        assert_eq!(outcome(&grep, &ToolOut::ok("a.rs"), ms).0, "1 match");
        assert_eq!(outcome(&grep, &ToolOut::ok("(no matches)"), ms).0, "no matches");
    }

    #[test]
    fn headless_writes_events_as_json_lines() {
        let mut h = Headless { out: Vec::new() };
        let e1 = Event { seq: 1, t_ms: 10, kind: Kind::User { text: "hi \"there\"\nline two".into() } };
        let e2 = Event { seq: 2, t_ms: 20, kind: Kind::SessionEnd { status: "done".into() } };
        h.event(&e1);
        h.event(&e2);

        // everything else is silent
        let c = call("read", r#"{"path":"a"}"#);
        h.text_delta("x");
        h.reasoning_delta("y");
        h.tool_start(&c);
        h.tool_end(&c, &ToolOut::ok("z"), 5);
        h.status(&Status::default());

        let text = std::str::from_utf8(&h.out).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), 2, "{text:?}");
        let parsed: Vec<Event> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(parsed, vec![e1, e2]);

        let req = PermissionRequest { tool: "bash".into(), summary: "ls".into(), detail: None, always: "ls".into() };
        assert_eq!(h.ask(&req), Decision::Deny);
        assert!(!h.confirm("continue?"));
    }

    #[test]
    fn ask_prompt_names_the_grant() {
        let req = |tool: &str, always: &str| PermissionRequest { tool: tool.into(), summary: String::new(), detail: None, always: always.into() };
        assert_eq!(ask_prompt(&req("bash", "cargo test, tail -20")), "    allow bash? [y]es [n]o [a]lways (cargo test, tail -20) › ");
        assert_eq!(ask_prompt(&req("edit", "edit")), "    allow edit? [y]es [n]o [a]lways (edit) › ");
        assert_eq!(ask_prompt(&req("bash", "")), "    allow bash? [y]es [n]o [a]lways › ");
    }
}

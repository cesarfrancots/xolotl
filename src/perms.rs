//! Per-call permission decisions: the three modes, "always" grants, `--allow` prefixes and the catastrophe list (spec §8).

use regex::RegexSet;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Ask,
    Writes,
    Yolo,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Ask => "ask",
            Mode::Writes => "writes",
            Mode::Yolo => "yolo",
        }
    }
}

impl FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "ask" => Ok(Mode::Ask),
            "writes" => Ok(Mode::Writes),
            "yolo" => Ok(Mode::Yolo),
            _ => Err(format!("unknown mode `{s}` (expected ask, writes or yolo)")),
        }
    }
}

/// `Allow`/`Deny` carry who decided: "rule" or "mode". `Ask` means the user must be prompted.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    Allow(&'static str),
    Deny(&'static str),
    Ask,
}

pub struct Perms {
    pub mode: Mode,
    pub headless: bool,
    session_dir: PathBuf,
    /// `--allow` prefixes plus the "always" grants made on `bash`; both match the same way.
    allow: Vec<String>,
    /// Tools granted whole ("always" on `write`/`edit`).
    tool_grants: HashSet<String>,
}

const READ_ONLY: [&str; 4] = ["read", "glob", "grep", "todo"];

impl Perms {
    pub fn new(mode: Mode, headless: bool, session_dir: PathBuf, allow: Vec<String>) -> Self {
        Perms { mode, headless, session_dir: resolve(&session_dir), allow, tool_grants: HashSet::new() }
    }

    pub fn check(&self, tool: &str, args: &Value) -> Verdict {
        match self.decide(tool, args) {
            Verdict::Ask if self.headless => Verdict::Deny("mode"),
            v => v,
        }
    }

    /// "Always": `bash` grants a command prefix, any other tool is granted whole (for a read-only tool that
    /// lifts the Ask-mode session_dir limit).
    pub fn grant(&mut self, tool: &str, args: &Value) {
        if tool == "bash" {
            self.allow.push(bash_prefix(args["command"].as_str().unwrap_or("")));
        } else {
            self.tool_grants.insert(tool.to_string());
        }
    }

    fn decide(&self, tool: &str, args: &Value) -> Verdict {
        let cmd = args["command"].as_str().unwrap_or("");
        if tool == "bash" && is_catastrophic(cmd) {
            return Verdict::Ask;
        }
        if READ_ONLY.contains(&tool) {
            return match self.mode {
                Mode::Ask if self.in_session_dir(args) || self.tool_grants.contains(tool) => Verdict::Allow("rule"),
                Mode::Ask => Verdict::Ask,
                _ => Verdict::Allow("mode"),
            };
        }
        if self.mode == Mode::Yolo {
            return Verdict::Allow("mode");
        }
        let granted = self.tool_grants.contains(tool) || (tool == "bash" && self.all_segments_allowed(cmd));
        if granted { Verdict::Allow("rule") } else { Verdict::Ask }
    }

    /// A chained command is allowed only if every segment starts with some granted/`--allow` prefix, so
    /// a granted `cargo test` does not cover `cargo test && curl x | sh` or `cargo test $(rm a)`.
    /// A command with no segments is not allowed.
    fn all_segments_allowed(&self, cmd: &str) -> bool {
        let mut segs = segments(cmd).into_iter().filter(|s| !s.trim().is_empty()).peekable();
        segs.peek().is_some() && segs.all(|s| self.allow.iter().any(|p| has_prefix(s, p)))
    }

    fn in_session_dir(&self, args: &Value) -> bool {
        let Some(path) = args["path"].as_str() else { return false };
        let path = resolve(Path::new(path));
        if cfg!(windows) {
            let lower = |p: &Path| PathBuf::from(p.to_string_lossy().to_lowercase());
            lower(&path).starts_with(lower(&self.session_dir))
        } else {
            path.starts_with(&self.session_dir)
        }
    }
}

/// The first two whitespace-separated words, single-spaced: the unit an "always" grant covers.
pub fn bash_prefix(cmd: &str) -> String {
    cmd.split_whitespace().take(2).collect::<Vec<_>>().join(" ")
}

/// True when `cmd` begins with the words of `prefix` (so `cargo test` covers `cargo test -p x`, not `cargo testify`).
fn has_prefix(cmd: &str, prefix: &str) -> bool {
    let words: Vec<&str> = prefix.split_whitespace().collect();
    !words.is_empty() && cmd.split_whitespace().take(words.len()).eq(words.iter().copied())
}

/// Absolute, lexically normalized (`.`/`..` folded) path; relative paths resolve against the process cwd.
// ponytail: symlinks are not resolved, so a link inside session_dir that points out still counts as inside.
// Upgrade path: canonicalize the deepest existing ancestor and re-append the rest.
fn resolve(p: &Path) -> PathBuf {
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

// ponytail: one heuristic regex per spec §8 bullet, run on each command segment. `rm`, `git push` and `del`/`rd`
// are matched anywhere in a segment after a boundary (space, `/`, `\`, quote), so `/bin/rm`, `\rm`, `FOO=1 rm`,
// `env rm`, `sudo -n rm`, `then rm` and `bash -c "rm .."` are all seen, at the cost of safe-direction false
// positives such as `echo rm -rf /`. `format` stays a command word so `echo format` is fine. Ceiling: this is
// not a shell parser, so variables (`$CMD -rf /`), aliases, `eval` of built strings and other quoting tricks
// slip past. Upgrade path: tokenize the shell command properly (or sandbox bash) instead of matching text.
static CATASTROPHES: LazyLock<RegexSet> = LazyLock::new(|| {
    let lead = r#"(?:^|[\s/\\"'])"#;
    // rm targets: `/`, `/*`, `~`, `$HOME`, a Git-Bash drive (`/c/`) or a Windows drive root (`C:\`), optionally quoted.
    let root = r#"["']?(?:/\*?|~/?\*?|\$HOME/?\*?|\$\{HOME\}/?\*?|/[a-z]/?\*?|[a-z]:[\\/]?\*?)["']?"#;
    let drive = r#"["']?[a-z]:[\\/]?\*?["']?"#;
    RegexSet::new([
        // rm with a recursive flag (-r, -R, -rf, -fr, -r -f, --recursive) and a root-like argument
        format!(r"(?i){lead}rm(?:\.exe)?\s+(?:\S+\s+)*?-(?:[a-z]*r[a-z]*|-recursive)\s+(?:\S+\s+)*?{root}(?:\s|$)"),
        // git push -f / --force (also catches --force-with-lease and bundled flags like -fu)
        format!(r"(?i){lead}git(?:\.exe)?\s+(?:\S+\s+)*?push\s+(?:\S+\s+)*?(?:-[a-z]*f[a-z]*|--force\S*)(?:\s|$)"),
        // format as the command word (not `--format=` or `fmt`)
        r"(?i)^\s*(?:sudo\s+)?format(?:\.com|\.exe)?(?:\s|$)".to_string(),
        // del/erase/rd/rmdir with /s and a drive root, flags in either order
        format!(r"(?i){lead}(?:del|erase|rd|rmdir)(?:\s+\S+)*?\s+(?:/s(?:\s+\S+)*?\s+{drive}|{drive}(?:\s+\S+)*?\s+/s)(?:\s|$)"),
    ])
    .expect("catastrophe patterns compile")
});

/// The commands of a shell line, split on `;`, `&&`, `||`, `|`, `&`, newline, backtick, `(` and `)` (empty pieces
/// are left in). The last three make `$(..)`, `<(..)`, `(..)` and `` `..` `` bodies segments of their own.
/// An `&` is a redirect (`2>&1`, `>&-`), not a separator, only right after an unescaped `>` and before a digit or `-`.
fn segments(cmd: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in cmd.char_indices() {
        let sep = match c {
            ';' | '|' | '\n' | '`' | '(' | ')' => true,
            '&' => {
                let (before, after) = (&cmd[..i], &cmd[i + 1..]);
                let redirect = before.ends_with('>') && !before.ends_with("\\>") && after.starts_with(|d: char| d.is_ascii_digit() || d == '-');
                !redirect
            }
            _ => false,
        };
        if sep {
            out.push(&cmd[start..i]);
            start = i + c.len_utf8();
        }
    }
    out.push(&cmd[start..]);
    out
}

/// True when any segment of `cmd` matches the catastrophe list, so `cd x && rm -rf /` and `echo $(rm -rf /)` are caught.
pub fn is_catastrophic(cmd: &str) -> bool {
    segments(cmd).into_iter().any(|seg| CATASTROPHES.is_match(seg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::test_dir;
    use serde_json::json;

    fn perms(mode: Mode, headless: bool, allow: &[&str]) -> (Perms, PathBuf) {
        let dir = test_dir("perms");
        (Perms::new(mode, headless, dir.clone(), allow.iter().map(|s| s.to_string()).collect()), dir)
    }
    fn bash(c: &str) -> Value {
        json!({ "command": c })
    }
    fn file(p: impl AsRef<Path>) -> Value {
        json!({ "path": p.as_ref().to_str().unwrap() })
    }

    #[test]
    fn mode_matrix() {
        for m in [Mode::Ask, Mode::Writes, Mode::Yolo] {
            assert_eq!(m.as_str().parse(), Ok(m));
        }
        assert!("loud".parse::<Mode>().is_err());

        // Writes: read/glob/grep/todo -> Allow("mode"); edit/write/bash -> Ask
        let (p, _) = perms(Mode::Writes, false, &[]);
        for t in ["read", "glob", "grep", "todo"] {
            assert_eq!(p.check(t, &json!({})), Verdict::Allow("mode"), "{t}");
        }
        for (t, a) in [("edit", file("a")), ("write", file("a")), ("bash", bash("ls"))] {
            assert_eq!(p.check(t, &a), Verdict::Ask, "{t}");
        }

        // Ask: reads are only free inside session_dir; a call with no path, and todo, ask.
        let (p, dir) = perms(Mode::Ask, false, &[]);
        let events = dir.join("events.jsonl");
        assert_eq!(p.check("read", &file(dir.parent().unwrap().join("other.txt"))), Verdict::Ask);
        assert_eq!(p.check("read", &file(&events)), Verdict::Allow("rule"));
        assert_eq!(p.check("grep", &json!({ "regex": "x", "path": dir.join("artifacts") })), Verdict::Allow("rule"));
        assert_eq!(p.check("glob", &json!({ "pattern": "*" })), Verdict::Ask);
        assert_eq!(p.check("todo", &json!({})), Verdict::Ask);
        assert_eq!(p.check("read", &file("Cargo.toml")), Verdict::Ask); // relative: resolved against the cwd, not session_dir
        assert_eq!(p.check("edit", &file(&events)), Verdict::Ask);
        assert_eq!(p.check("bash", &bash("ls")), Verdict::Ask);
        // `..` escapes and a sibling that merely shares the prefix are outside.
        assert_eq!(p.check("read", &file(dir.join("..").join("x"))), Verdict::Ask);
        assert_eq!(p.check("read", &file(PathBuf::from(format!("{}-evil", dir.display())).join("x"))), Verdict::Ask);
        assert_eq!(p.check("read", &file(dir.join("a").join("..").join("events.jsonl"))), Verdict::Allow("rule"));
        #[cfg(windows)]
        assert_eq!(p.check("read", &file(events.to_str().unwrap().to_uppercase())), Verdict::Allow("rule"));

        // Yolo: everything gated is allowed.
        let (p, _) = perms(Mode::Yolo, false, &[]);
        assert_eq!(p.check("edit", &file("a")), Verdict::Allow("mode"));
        assert_eq!(p.check("bash", &bash("ls")), Verdict::Allow("mode"));
    }

    #[test]
    fn grants_and_prefixes() {
        assert_eq!(bash_prefix("cargo test --all"), "cargo test");
        assert_eq!(bash_prefix("  ls   -la x"), "ls -la");
        assert_eq!(bash_prefix("make"), "make");

        // grant("edit") -> edit Allow("rule"), write still Ask
        let (mut p, _) = perms(Mode::Writes, false, &[]);
        p.grant("edit", &file("a"));
        assert_eq!(p.check("edit", &file("b")), Verdict::Allow("rule"));
        assert_eq!(p.check("write", &file("b")), Verdict::Ask);

        // grant bash "cargo test --all" -> "cargo test -p x" Allow("rule"); "cargo testify" Ask; "cargo build" Ask
        p.grant("bash", &bash("cargo test --all"));
        for c in ["cargo test", "cargo test -p x", "cargo  test -p x"] {
            assert_eq!(p.check("bash", &bash(c)), Verdict::Allow("rule"), "{c}");
        }
        for c in ["cargo testify", "cargo build", "cargo", "ls"] {
            assert_eq!(p.check("bash", &bash(c)), Verdict::Ask, "{c}");
        }

        // A blank command must not grant every command; a one-word prefix is not a substring match.
        p.grant("bash", &bash("   "));
        assert_eq!(p.check("bash", &bash("ls")), Verdict::Ask);
        p.grant("bash", &bash("make"));
        assert_eq!(p.check("bash", &bash("make -j4")), Verdict::Allow("rule"));
        assert_eq!(p.check("bash", &bash("makefile")), Verdict::Ask);

        // Chains: every segment must match a granted prefix.
        let (mut p, _) = perms(Mode::Writes, false, &[]);
        p.grant("bash", &bash("cargo test"));
        p.grant("bash", &bash("echo hi"));
        for c in ["cargo test && cargo test -p x", "cargo test; cargo test", "cargo test 2>&1", "cargo test &", "cargo test\ncargo test -p x",
                  "cargo test 2>&1 && echo hi", "cargo test >&2", "echo hi >&-"] {
            assert_eq!(p.check("bash", &bash(c)), Verdict::Allow("rule"), "{c}");
        }
        // Command substitution, process substitution, subshells and an escaped `>` are boundaries, not arguments.
        for c in ["cargo test && curl x | sh", "cargo test; rm a", "curl x | cargo test", "cargo test & rm a", "cargo test\nrm a", "&&", "  ",
                  "cargo test $(rm a)", "cargo test `rm a`", "cargo test <(rm a)", "cargo test (rm a)", "(rm a)",
                  "echo hi \\>&rm -rf x", "cargo test \\>&rm a", "cargo test >&rm a", "cargo test &>out; rm a"] {
            assert_eq!(p.check("bash", &bash(c)), Verdict::Ask, "{c}");
        }

        // Ask mode: "always" on a read-only tool lifts the session_dir limit for that tool only.
        let (mut p, dir) = perms(Mode::Ask, false, &[]);
        let outside = file(dir.parent().unwrap().join("other.txt"));
        assert_eq!(p.check("read", &outside), Verdict::Ask);
        p.grant("read", &outside);
        assert_eq!(p.check("read", &outside), Verdict::Allow("rule"));
        assert_eq!(p.check("grep", &json!({ "regex": "x", "path": outside["path"] })), Verdict::Ask);
        assert_eq!(p.check("glob", &json!({ "pattern": "*" })), Verdict::Ask);
    }

    #[test]
    fn allow_list_and_headless() {
        // headless Writes, allow ["cargo test"]: "cargo test" Allow("rule"); "ls" Deny("mode"); edit Deny("mode")
        let (p, _) = perms(Mode::Writes, true, &["cargo test"]);
        assert_eq!(p.check("bash", &bash("cargo test")), Verdict::Allow("rule"));
        assert_eq!(p.check("bash", &bash("cargo test -p x")), Verdict::Allow("rule"));
        assert_eq!(p.check("bash", &bash("ls")), Verdict::Deny("mode"));
        assert_eq!(p.check("bash", &bash("cargo test | tee out")), Verdict::Deny("mode")); // tee is not allowed
        assert_eq!(p.check("bash", &bash("cargo test && cargo test -p x")), Verdict::Allow("rule"));
        assert_eq!(p.check("edit", &file("a")), Verdict::Deny("mode"));
        assert_eq!(p.check("read", &file("a")), Verdict::Allow("mode"));

        // headless Yolo: edit Allow("mode")
        let (p, _) = perms(Mode::Yolo, true, &[]);
        assert_eq!(p.check("edit", &file("a")), Verdict::Allow("mode"));

        // Interactive, the same allow-list is just a rule.
        let (p, _) = perms(Mode::Writes, false, &["cargo test"]);
        assert_eq!(p.check("bash", &bash("cargo test")), Verdict::Allow("rule"));
        assert_eq!(p.check("bash", &bash("ls")), Verdict::Ask);
    }

    #[test]
    fn catastrophes_always_ask() {
        let bad = ["rm -rf /", "cd x && rm -rf /", "ls; git push --force", "make | rm -rf ~", "git push -f origin main"];
        // Yolo: is_catastrophic commands -> Ask; headless Yolo -> Deny("mode")
        let (mut p, _) = perms(Mode::Yolo, false, &[]);
        assert_eq!(p.check("bash", &bash("ls")), Verdict::Allow("mode"));
        for c in bad {
            assert!(is_catastrophic(c), "{c}");
            assert_eq!(p.check("bash", &bash(c)), Verdict::Ask, "{c}");
        }
        p.headless = true;
        for c in bad {
            assert_eq!(p.check("bash", &bash(c)), Verdict::Deny("mode"), "{c}");
        }

        // Wrappers, paths, subshells, substitutions and control-flow keywords do not hide a catastrophe (Yolo still asks).
        let wrapped = ["/bin/rm -rf /", "\\rm -rf /", "FOO=1 rm -rf /", "command rm -rf /", "env rm -rf /", "time rm -rf /",
                       "sudo -n rm -rf /", "(rm -rf /)", "{ rm -rf /; }", "if true; then rm -rf /; fi", "for x in 1; do rm -rf /; done",
                       "echo $(rm -rf /)", "echo `rm -rf /`", "echo \\>&rm -rf /", "/usr/bin/git push -f", "git.exe push -f",
                       "rm.exe -rf /", "bash -c \"rm -rf /\"", "sh -c 'cd x && rm -rf ~'", "cmd /c \"rd /s /q C:\\\"", "xargs rm -rf /"];
        let (p, _) = perms(Mode::Yolo, false, &[]);
        for c in wrapped {
            assert!(is_catastrophic(c), "{c}");
            assert_eq!(p.check("bash", &bash(c)), Verdict::Ask, "{c}");
        }
        // `format` stays a command word, so mentioning it is fine.
        for c in ["echo format C:", "echo \"format C:\"", "ls /bin/format-all"] {
            assert!(!is_catastrophic(c), "{c}");
        }

        // Neither a grant nor an --allow prefix lifts a catastrophe.
        let (mut p, _) = perms(Mode::Writes, false, &["git push"]);
        p.grant("bash", &bash("rm -rf build"));
        assert_eq!(p.check("bash", &bash("rm -rf build")), Verdict::Allow("rule"));
        assert_eq!(p.check("bash", &bash("rm -rf /")), Verdict::Ask);
        assert_eq!(p.check("bash", &bash("git push --force")), Verdict::Ask);
        assert_eq!(p.check("bash", &bash("git push origin main")), Verdict::Allow("rule"));
    }

    #[test]
    fn catastrophe_table() {
        for c in ["rm -rf /", "rm -fr ~", "rm -rf $HOME", "rm -rf /c/", "rm -rf C:\\", "rm -r -f /*",
                  "git push --force origin main", "git push -f", "format C:", "del /s /q C:\\", "rd /s /q D:\\"] {
            assert!(is_catastrophic(c), "{c}");
        }
        for c in ["rm -rf ./build", "rm -rf target", "git push origin main", "git log --format=%H",
                  "cargo fmt", "del notes.txt"] {
            assert!(!is_catastrophic(c), "{c}");
        }
    }
}

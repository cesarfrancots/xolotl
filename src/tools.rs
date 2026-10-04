//! File tools `read`, `write`, `edit` (spec §7, §7.1), shell selection and the `bash` tool (§7.2), and the
//! char-boundary-safe `clip` helpers.

use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const MAX_READ_LINES: usize = 2000;
const MAX_LINE_BYTES: usize = 2000;
const BINARY_SNIFF_BYTES: usize = 8192;

/// How often `exec_shell` checks the child, the cancel flag and the deadline.
const POLL: Duration = Duration::from_millis(50);
/// After the child exits or is killed, how long to wait for its output pipes to close (a background process
/// may hold them open for much longer).
const DRAIN: Duration = Duration::from_secs(2);
/// Captured output is capped so a runaway command (`yes`) cannot exhaust memory.
const MAX_CAPTURE: usize = 32 << 20;

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq)]
pub enum Shell {
    Bash(PathBuf),
    Sh,
    PowerShell,
}

pub struct ToolCtx {
    pub root: PathBuf,
    pub shell: Shell,
    pub backups_dir: PathBuf,
    /// Files read this session, keyed by `fs::canonicalize(path)`, with the mtime seen at read time.
    pub reads: HashMap<PathBuf, SystemTime>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolOut {
    pub ok: bool,
    pub text: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
}

impl ToolOut {
    pub fn ok(text: impl Into<String>) -> Self {
        ToolOut { ok: true, text: text.into(), exit_code: None, cancelled: false }
    }
    pub fn err(text: impl Into<String>) -> Self {
        ToolOut { ok: false, ..Self::ok(text) }
    }
}

/// Longest prefix of `s` that is at most `max_bytes` long and ends on a char boundary.
pub fn clip(s: &str, max_bytes: usize) -> &str {
    &s[..s.floor_char_boundary(max_bytes)]
}

/// Longest suffix of `s` that is at most `max_bytes` long and starts on a char boundary.
pub fn clip_tail(s: &str, max_bytes: usize) -> &str {
    &s[s.ceil_char_boundary(s.len().saturating_sub(max_bytes))..]
}

pub(crate) fn read(ctx: &mut ToolCtx, args: &Value) -> ToolOut {
    done(read_impl(ctx, args))
}

pub(crate) fn write(ctx: &mut ToolCtx, call_id: &str, args: &Value) -> ToolOut {
    done(write_impl(ctx, call_id, args))
}

pub(crate) fn edit(ctx: &mut ToolCtx, call_id: &str, args: &Value) -> ToolOut {
    done(edit_impl(ctx, call_id, args))
}

fn done(r: Result<String, String>) -> ToolOut {
    match r {
        Ok(text) => ToolOut::ok(text),
        Err(text) => ToolOut::err(text),
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| format!("missing argument `{key}`"))
}

/// Optional argument: absent or null is `None`; present with the wrong type is an error.
fn opt_arg<T>(args: &Value, key: &str, get: fn(&Value) -> Option<T>, want: &str) -> Result<Option<T>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => get(v).map(Some).ok_or_else(|| format!("argument `{key}` must be {want}")),
    }
}

fn io_err(path: &str, e: &io::Error) -> String {
    if e.kind() == io::ErrorKind::NotFound {
        format!("file not found: {path}")
    } else {
        format!("cannot access {path}: {e}")
    }
}

/// `line 2` / `lines 1, 2` (first 10 shown).
fn lines_label(at: &[usize]) -> String {
    let shown = at.iter().take(10).map(usize::to_string).collect::<Vec<_>>().join(", ");
    format!("line{} {shown}{}", if at.len() == 1 { "" } else { "s" }, if at.len() > 10 { ", …" } else { "" })
}

/// Errors unless `canon` was read this session and its mtime is unchanged since.
fn check_read(ctx: &ToolCtx, canon: &Path, shown: &str) -> Result<(), String> {
    let Some(&seen) = ctx.reads.get(canon) else {
        return Err(format!("{shown} was not read this session — read it first"));
    };
    match fs::metadata(canon).and_then(|m| m.modified()) {
        Ok(now) if now == seen => Ok(()),
        _ => Err("file changed since read — read it again".into()),
    }
}

/// Backs up the existing file (if any) to `backups_dir/<call_id>.before`, writes `bytes` to a temp file next
/// to `target`, renames it over `target`, and refreshes `ctx.reads` so the next write/edit needs no re-read.
fn commit(ctx: &mut ToolCtx, call_id: &str, target: &Path, bytes: &[u8]) -> io::Result<()> {
    let perms = fs::metadata(target).ok().map(|m| m.permissions());
    if perms.is_some() {
        // call ids come from the model: keep only filename-safe characters
        let id: String = call_id.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '-' }).collect();
        let backup = fs::create_dir_all(&ctx.backups_dir).and_then(|()| fs::copy(target, ctx.backups_dir.join(format!("{id}.before"))));
        backup.map_err(|e| io::Error::new(e.kind(), format!("backup failed: {e}")))?;
    }
    if let Some(dir) = target.parent() {
        fs::create_dir_all(dir)?;
    }
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = target.with_file_name(format!(".{name}.xolotl-tmp"));
    fs::write(&tmp, bytes)?;
    if let Err(e) = fs::rename(&tmp, target) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    if let Some(p) = perms {
        let _ = fs::set_permissions(target, p); // the temp file is born with default perms (e.g. loses +x)
    }
    if let (Ok(canon), Ok(mtime)) = (fs::canonicalize(target), fs::metadata(target).and_then(|m| m.modified())) {
        ctx.reads.insert(canon, mtime);
    }
    Ok(())
}

fn read_impl(ctx: &mut ToolCtx, args: &Value) -> Result<String, String> {
    let path = str_arg(args, "path")?;
    let offset = opt_arg(args, "offset", Value::as_u64, "a non-negative integer")?.unwrap_or(1).max(1) as usize;
    let limit = opt_arg(args, "limit", Value::as_u64, "a non-negative integer")?.unwrap_or(MAX_READ_LINES as u64).max(1) as usize;

    let p = ctx.root.join(path);
    let meta = fs::metadata(&p).map_err(|e| io_err(path, &e))?;
    if meta.is_dir() {
        return Err(format!("{path} is a directory"));
    }
    let bytes = fs::read(&p).map_err(|e| io_err(path, &e))?;
    if bytes.iter().take(BINARY_SNIFF_BYTES).any(|&b| b == 0) {
        return Ok(format!("binary file, {} bytes", bytes.len()));
    }
    // mtime comes from the metadata fetched before the read: a racing change can only make it look stale
    if let (Ok(canon), Ok(mtime)) = (fs::canonicalize(&p), meta.modified()) {
        ctx.reads.insert(canon, mtime);
    }

    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok("(empty file)".into());
    }
    let start = offset - 1;
    if start >= lines.len() {
        return Err(format!("offset {offset} is past the end of {path} ({} lines)", lines.len()));
    }
    let end = start.saturating_add(limit).min(lines.len());
    let mut out = lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let kept = clip(line, MAX_LINE_BYTES);
            let cut = if kept.len() < line.len() { " …[line cut]" } else { "" };
            format!("{:>6}\t{kept}{cut}", start + i + 1)
        })
        .collect::<Vec<_>>()
        .join("\n");
    if end < lines.len() {
        out += &format!("\n[{} more lines — continue with offset={}]", lines.len() - end, end + 1);
    }
    Ok(out)
}

fn write_impl(ctx: &mut ToolCtx, call_id: &str, args: &Value) -> Result<String, String> {
    let path = str_arg(args, "path")?;
    let content = str_arg(args, "content")?;
    let p = ctx.root.join(path);
    // Existing files resolve to their real path (so a symlink is written through, not replaced).
    let target = match fs::canonicalize(&p) {
        Ok(canon) => {
            check_read(ctx, &canon, path)?;
            canon
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => p,
        Err(e) => return Err(io_err(path, &e)),
    };
    commit(ctx, call_id, &target, content.as_bytes()).map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(format!("wrote {path} ({} lines)", content.lines().count()))
}

fn edit_impl(ctx: &mut ToolCtx, call_id: &str, args: &Value) -> Result<String, String> {
    let path = str_arg(args, "path")?;
    let old = str_arg(args, "old")?;
    let new = str_arg(args, "new")?;
    let replace_all = opt_arg(args, "replace_all", Value::as_bool, "a boolean")?.unwrap_or(false);
    if old.is_empty() {
        return Err("argument `old` must not be empty".into());
    }

    let target = fs::canonicalize(ctx.root.join(path)).map_err(|e| io_err(path, &e))?;
    check_read(ctx, &target, path)?;
    let bytes = fs::read(&target).map_err(|e| io_err(path, &e))?;
    let content = String::from_utf8(bytes).map_err(|_| format!("{path} is not valid UTF-8; edit only supports UTF-8 text"))?;

    // The file's line endings win: model-supplied text is normalised to LF, then converted if the file is CRLF.
    let eol = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let norm = |s: &str| s.replace("\r\n", "\n").replace('\n', eol);
    let (old_n, new_n) = (norm(old), norm(new));

    let hits: Vec<usize> = content.match_indices(&old_n).map(|(i, _)| i).collect();
    let (updated, count, note) = if hits.is_empty() {
        // ponytail: fuzzy match is trimmed-line equality only; a plain scan of every window (O(file lines × old lines)).
        // Upgrade path if it bites: normalise internal whitespace runs too.
        let segs: Vec<&str> = content.split_inclusive('\n').collect();
        let want: Vec<&str> = old_n.lines().map(str::trim).collect();
        let found: Vec<usize> = segs
            .windows(want.len().max(1))
            .enumerate()
            .filter(|(_, win)| win.iter().zip(&want).all(|(seg, w)| seg.trim() == *w))
            .map(|(i, _)| i)
            .collect();
        match found[..] {
            [i] => {
                let j = i + want.len(); // window is segs[i..j]
                let new_lines: Vec<&str> = new_n.lines().collect();
                let mut rep = new_lines.join(eol);
                if !new_lines.is_empty() && segs[j - 1].ends_with('\n') {
                    rep += eol; // keep the terminator of the last replaced line
                }
                let updated = [segs[..i].concat(), rep, segs[j..].concat()].concat();
                (updated, 1, format!(", applied with whitespace-insensitive match at lines {}–{j}", i + 1))
            }
            [] => {
                let first = old_n.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
                let at: Vec<usize> = if first.is_empty() {
                    vec![]
                } else {
                    content.lines().enumerate().filter(|(_, l)| l.contains(first)).map(|(i, _)| i + 1).collect()
                };
                let shown = clip(first, 80);
                return Err(if at.is_empty() {
                    format!("`old` not found in {path}: no line contains `{shown}`")
                } else {
                    format!(
                        "`old` not found in {path}: its first line `{shown}` appears at {}, but the lines after it differ — read the file again and copy the text exactly",
                        lines_label(&at)
                    )
                });
            }
            _ => {
                let at: Vec<usize> = found.iter().map(|i| i + 1).collect();
                return Err(format!(
                    "{} whitespace-insensitive matches at {} — add surrounding context to make `old` unique",
                    found.len(),
                    lines_label(&at)
                ));
            }
        }
    } else if hits.len() > 1 && !replace_all {
        let at: Vec<usize> = hits.iter().map(|&i| 1 + content.as_bytes()[..i].iter().filter(|&&b| b == b'\n').count()).collect();
        return Err(format!(
            "{} matches at {} — add surrounding context to make `old` unique, or set replace_all",
            hits.len(),
            lines_label(&at)
        ));
    } else {
        let note = if replace_all { format!(", replaced {} occurrence{}", hits.len(), if hits.len() == 1 { "" } else { "s" }) } else { String::new() };
        (content.replace(&old_n, &new_n), hits.len(), note)
    };

    commit(ctx, call_id, &target, updated.as_bytes()).map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(format!("edited {path} (+{} −{}){note}", new.lines().count() * count, old.lines().count() * count))
}

// ---- shell (spec §7.2) ----

/// Windows: Git Bash (`<git>\..\..\bin\bash.exe`, then `extra` in order; never the WSL launcher at
/// `System32\bash.exe`), else PowerShell. Elsewhere: `bash` if it is installed, else `sh`.
pub fn resolve_shell_from(windows: bool, git_exe: Option<&Path>, extra: &[PathBuf], exists: &dyn Fn(&Path) -> bool) -> Shell {
    if !windows {
        let has_bash = ["/bin/bash", "/usr/bin/bash"].iter().any(|p| exists(Path::new(p)));
        return if has_bash { Shell::Bash(PathBuf::from("bash")) } else { Shell::Sh };
    }
    let from_git = git_exe.and_then(Path::parent).and_then(Path::parent).map(|root| root.join("bin").join("bash.exe"));
    from_git.iter().chain(extra).find(|p| exists(p) && !is_wsl_launcher(p)).map_or(Shell::PowerShell, |p| Shell::Bash(p.clone()))
}

fn is_wsl_launcher(p: &Path) -> bool {
    p.to_string_lossy().replace('/', "\\").to_lowercase().ends_with(r"\system32\bash.exe")
}

pub fn resolve_shell() -> Shell {
    let git = if cfg!(windows) { where_git() } else { None };
    let mut extra = vec![PathBuf::from(r"C:\Program Files\Git\bin\bash.exe"), PathBuf::from(r"C:\Program Files (x86)\Git\bin\bash.exe")];
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        extra.push(PathBuf::from(local).join(r"Programs\Git\bin\bash.exe"));
    }
    resolve_shell_from(cfg!(windows), git.as_deref(), &extra, &|p| p.is_file())
}

/// First line of `where git`.
fn where_git() -> Option<PathBuf> {
    let out = Command::new("where").arg("git").stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    let line = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    (!line.is_empty()).then(|| PathBuf::from(line))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Exec {
    /// Combined stdout and stderr, in arrival order.
    pub output: String,
    /// `None` when the process was killed (timeout, cancel), died on a signal, or never started.
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub duration_ms: u64,
}

fn shell_command(shell: &Shell, cmd: &str) -> Command {
    match shell {
        Shell::Bash(path) => bash_command(path, cmd),
        Shell::Sh => {
            let mut c = Command::new("sh");
            c.args(["-c", cmd]);
            c
        }
        Shell::PowerShell => {
            let mut c = Command::new("powershell");
            c.args(["-NoProfile", "-NonInteractive", "-Command", cmd]);
            c
        }
    }
}

#[cfg(not(windows))]
fn bash_command(path: &Path, cmd: &str) -> Command {
    let mut c = Command::new(path);
    c.args(["-lc", cmd]);
    c
}

/// Git Bash's runtime (msys) splits its command line differently from the Windows rules `Command::arg` quotes
/// for: anywhere in an argument, a backslash before a backslash or a double quote is an escape. `echo 'a\\b'`
/// would reach bash as `echo 'a\b'`. So the command is quoted by hand, escaping every backslash and quote.
#[cfg(windows)]
fn bash_command(path: &Path, cmd: &str) -> Command {
    use std::os::windows::process::CommandExt;
    let mut c = Command::new(path);
    let quoted = format!("\"{}\"", cmd.replace('\\', r"\\").replace('"', r#"\""#));
    c.arg("-c").raw_arg(quoted).env("CHERE_INVOKING", "1"); // CHERE_INVOKING: stay in the inherited cwd
    c
}

/// Forwards everything `pipe` yields to `tx` as byte chunks; `tx` is dropped when the pipe closes.
fn spawn_reader<R: Read + Send + 'static>(pipe: Option<R>, tx: mpsc::Sender<Vec<u8>>) {
    let Some(mut pipe) = pipe else { return };
    thread::spawn(move || {
        let mut buf = [0u8; 16 * 1024];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(buf[..n].to_vec()).is_err() {
                        break; // exec_shell has returned
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
    });
}

/// Appends `chunk` to `out` up to `MAX_CAPTURE` bytes in total, noting in `truncated` that the rest was dropped.
fn keep(out: &mut Vec<u8>, truncated: &mut bool, chunk: &[u8]) {
    let room = MAX_CAPTURE.saturating_sub(out.len());
    *truncated |= chunk.len() > room;
    out.extend_from_slice(&chunk[..chunk.len().min(room)]);
}

/// Kills `child` and everything it started, then reaps it.
fn kill_tree(child: &mut Child) {
    let pid = child.id().to_string();
    let mut killer = if cfg!(windows) {
        let mut c = Command::new("taskkill");
        c.args(["/T", "/F", "/PID", &pid]);
        c
    } else {
        let mut c = Command::new("kill"); // the child leads its own process group (see `exec_shell`)
        c.args(["-KILL", &format!("-{pid}")]);
        c
    };
    let _ = killer.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    let _ = child.kill(); // fallback when the tool above is missing or refused: at least stop the shell itself
    let _ = child.wait();
}

/// Runs `cmd` in a fresh `shell` process in `cwd` (stdin closed) until it exits, `timeout` passes, or `cancel`
/// is set; the last two kill the whole process tree and keep the partial output.
pub fn exec_shell(shell: &Shell, cmd: &str, cwd: &Path, timeout: Duration, cancel: &AtomicBool) -> Exec {
    let start = Instant::now();
    let elapsed_ms = || u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let mut command = shell_command(shell, cmd);
    command
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_EDITOR", "false")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            let output = format!("failed to start {}: {e}", command.get_program().to_string_lossy());
            return Exec { output, exit_code: None, timed_out: false, cancelled: false, duration_ms: elapsed_ms() };
        }
    };

    let (tx, rx) = mpsc::channel();
    spawn_reader(child.stdout.take(), tx.clone());
    spawn_reader(child.stderr.take(), tx); // `rx` disconnects once both readers are done

    let deadline = start + timeout;
    let (mut out, mut truncated) = (Vec::new(), false);
    let mut readers_open = true;
    let (mut timed_out, mut cancelled) = (false, false);
    let status = loop {
        if readers_open {
            match rx.recv_timeout(POLL) {
                Ok(chunk) => keep(&mut out, &mut truncated, &chunk),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => readers_open = false,
            }
        } else {
            thread::sleep(POLL);
        }
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if cancel.load(Ordering::SeqCst) => cancelled = true,
            Ok(None) if Instant::now() >= deadline => timed_out = true,
            Ok(None) => continue,
            Err(_) => {} // cannot observe the child any more: kill it rather than wait forever
        }
        kill_tree(&mut child);
        break None;
    };

    // Collect what the readers still hold, but never wait for them past DRAIN and never join them: a
    // background process may keep the pipes open for as long as it lives.
    if readers_open {
        let end = Instant::now() + DRAIN;
        while let Some(left) = end.checked_duration_since(Instant::now()) {
            match rx.recv_timeout(left) {
                Ok(chunk) => keep(&mut out, &mut truncated, &chunk),
                Err(_) => break,
            }
        }
    }
    let mut output = String::from_utf8_lossy(&out).into_owned();
    if truncated {
        // ponytail: keeps only the head of a runaway output. Upgrade path: spill the rest to a file.
        output += &format!("\n[output limit reached: {MAX_CAPTURE} bytes kept, the rest was discarded]");
    }
    Exec { output, exit_code: status.and_then(|s| s.code()), timed_out, cancelled, duration_ms: elapsed_ms() }
}

/// `timeout` seconds, default 120, clamped to 1..=600 (fractions are dropped).
pub(crate) fn bash_timeout(args: &Value) -> Duration {
    let secs = args.get("timeout").and_then(Value::as_f64).unwrap_or(120.0);
    Duration::from_secs(secs.clamp(1.0, 600.0) as u64)
}

pub(crate) fn bash(ctx: &mut ToolCtx, args: &Value, cancel: &AtomicBool) -> ToolOut {
    let cmd = match str_arg(args, "command") {
        Ok(cmd) => cmd,
        Err(e) => return ToolOut::err(e),
    };
    let timeout = bash_timeout(args);
    let e = exec_shell(&ctx.shell, cmd, &ctx.root, timeout, cancel);
    let secs = e.duration_ms as f64 / 1000.0;
    let footer = match e.exit_code {
        _ if e.cancelled => "\n[cancelled by user]".to_string(),
        _ if e.timed_out => format!("\n[timed out after {}s · partial output above]", timeout.as_secs()),
        Some(code) => format!("\n[exit {code} · {secs:.1}s]"),
        None => format!("\n[terminated without an exit code · {secs:.1}s]"),
    };
    ToolOut { ok: e.exit_code == Some(0), text: e.output + &footer, exit_code: e.exit_code, cancelled: e.cancelled }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::test_dir;
    use serde_json::json;
    use std::time::Duration;

    fn ctx(tag: &str) -> ToolCtx {
        let root = test_dir(tag);
        ToolCtx { backups_dir: root.join(".bk"), root, shell: Shell::Sh, reads: HashMap::new() }
    }
    fn put(c: &ToolCtx, name: &str, bytes: impl AsRef<[u8]>) {
        fs::write(c.root.join(name), bytes).unwrap();
    }
    fn get(c: &ToolCtx, name: &str) -> Vec<u8> {
        fs::read(c.root.join(name)).unwrap()
    }
    fn gets(c: &ToolCtx, name: &str) -> String {
        String::from_utf8(get(c, name)).unwrap()
    }
    fn rd(c: &mut ToolCtx, name: &str) {
        assert!(read(c, &json!({ "path": name })).ok);
    }
    fn ed(c: &mut ToolCtx, id: &str, name: &str, old: &str, new: &str) -> ToolOut {
        edit(c, id, &json!({ "path": name, "old": old, "new": new }))
    }

    #[test]
    fn clip_never_splits_chars() {
        assert_eq!(clip("aé", 2), "a");
        assert_eq!(clip_tail("éa", 2), "a");
        assert_eq!(clip("──", 4), "─");
        assert_eq!(clip("abc", 10), "abc");
        assert_eq!(clip_tail("abc", 10), "abc");
    }

    #[test]
    fn read_numbers_lines_with_offset_and_limit() {
        let mut c = ctx("read-offset");
        put(&c, "f.txt", "l1\nl2\nl3\n");
        let o = read(&mut c, &json!({ "path": "f.txt", "offset": 2, "limit": 1 }));
        assert!(o.ok, "{}", o.text);
        assert_eq!(o.text, "     2\tl2\n[1 more lines — continue with offset=3]");
        // absolute paths are used as-is; no footer when everything fits
        let abs = c.root.join("f.txt").to_string_lossy().into_owned();
        let o = read(&mut c, &json!({ "path": abs }));
        assert_eq!(o.text, "     1\tl1\n     2\tl2\n     3\tl3");
    }

    #[test]
    fn read_cuts_long_lines_and_detects_binary() {
        let mut c = ctx("read-edge");
        put(&c, "long.txt", format!("{}\n", "x".repeat(2500)));
        let o = read(&mut c, &json!({ "path": "long.txt" }));
        assert!(o.ok);
        assert_eq!(o.text, format!("     1\t{} …[line cut]", "x".repeat(2000)));

        put(&c, "bin", [0u8, 1, 2]);
        let o = read(&mut c, &json!({ "path": "bin" }));
        assert!(o.ok);
        assert_eq!(o.text, "binary file, 3 bytes");

        let o = read(&mut c, &json!({ "path": "nope.txt" }));
        assert!(!o.ok);
        assert!(o.text.starts_with("file not found"), "{}", o.text);
    }

    #[test]
    fn write_requires_prior_read_and_backs_up() {
        let mut c = ctx("write");
        let o = write(&mut c, "w1", &json!({ "path": "d/n.txt", "content": "new" }));
        assert!(o.ok, "{}", o.text);
        assert_eq!(gets(&c, "d/n.txt"), "new");

        put(&c, "e.txt", "old");
        let o = write(&mut c, "w2", &json!({ "path": "e.txt", "content": "x" }));
        assert!(!o.ok && o.text.contains("read it first"), "{}", o.text);
        assert_eq!(gets(&c, "e.txt"), "old");

        rd(&mut c, "e.txt");
        let o = write(&mut c, "w3", &json!({ "path": "e.txt", "content": "v1" }));
        assert!(o.ok, "{}", o.text);
        assert_eq!(gets(&c, "e.txt"), "v1");
        assert_eq!(fs::read_to_string(c.backups_dir.join("w3.before")).unwrap(), "old");

        // reads map was refreshed by the previous write: no re-read needed
        let o = write(&mut c, "w4", &json!({ "path": "e.txt", "content": "v2" }));
        assert!(o.ok, "{}", o.text);
        assert_eq!(gets(&c, "e.txt"), "v2");
        assert_eq!(fs::read_to_string(c.backups_dir.join("w4.before")).unwrap(), "v1");
        assert!(!c.root.join(".e.txt.xolotl-tmp").exists());

        // call ids come from the model: unsafe characters never reach the filename
        let o = write(&mut c, "../a b", &json!({ "path": "e.txt", "content": "v3" }));
        assert!(o.ok, "{}", o.text);
        assert_eq!(fs::read_to_string(c.backups_dir.join("..-a-b.before")).unwrap(), "v2");
    }

    #[test]
    fn edit_unique_ambiguous_and_replace_all() {
        let mut c = ctx("edit-match");
        put(&c, "a.rs", "fn a() {}\nfn b() {}\n");
        rd(&mut c, "a.rs");
        let o = ed(&mut c, "e1", "a.rs", "fn b() {}", "fn b() { 1 }");
        assert!(o.ok, "{}", o.text);
        assert_eq!(gets(&c, "a.rs"), "fn a() {}\nfn b() { 1 }\n");

        put(&c, "x.txt", "x\nx\n");
        rd(&mut c, "x.txt");
        let o = ed(&mut c, "e2", "x.txt", "x", "y");
        assert!(!o.ok, "{}", o.text);
        assert!(o.text.contains("2 matches") && o.text.contains("lines 1, 2"), "{}", o.text);
        assert_eq!(gets(&c, "x.txt"), "x\nx\n");

        let o = edit(&mut c, "e3", &json!({ "path": "x.txt", "old": "x", "new": "y", "replace_all": true }));
        assert!(o.ok && o.text.contains("2 occurrences"), "{}", o.text);
        assert_eq!(gets(&c, "x.txt"), "y\ny\n");
    }

    #[test]
    fn edit_whitespace_fallback() {
        let mut c = ctx("edit-fuzzy");
        put(&c, "a.rs", "    let a = 1;\n    let b = 2;\n");
        rd(&mut c, "a.rs");
        let o = ed(&mut c, "e1", "a.rs", "let a = 1;\nlet b = 2;", "    let a = 10;\n    let b = 2;");
        assert!(o.ok, "{}", o.text);
        assert_eq!(gets(&c, "a.rs"), "    let a = 10;\n    let b = 2;\n");
        assert!(o.text.contains("whitespace-insensitive match at lines 1–2"), "{}", o.text);
    }

    #[test]
    fn edit_no_match_hints() {
        let mut c = ctx("edit-hints");
        put(&c, "a.txt", "alpha\nbeta\n");
        rd(&mut c, "a.txt");
        let o = ed(&mut c, "e1", "a.txt", "beta\nzzz", "q");
        assert!(!o.ok && o.text.contains("line 2"), "{}", o.text);
        let o = ed(&mut c, "e2", "a.txt", "gamma", "q");
        assert!(!o.ok && o.text.contains("no line contains"), "{}", o.text);
        assert_eq!(gets(&c, "a.txt"), "alpha\nbeta\n");
    }

    #[test]
    fn edit_preserves_crlf() {
        let mut c = ctx("edit-crlf");
        put(&c, "w.txt", "a\r\nb\r\n");
        rd(&mut c, "w.txt");
        let o = ed(&mut c, "e1", "w.txt", "a\nb", "x\ny");
        assert!(o.ok, "{}", o.text);
        assert_eq!(get(&c, "w.txt"), b"x\r\ny\r\n");
    }

    #[test]
    fn edit_rejects_stale_and_unread() {
        let mut c = ctx("edit-stale");
        put(&c, "a.txt", "hello\n");
        let o = ed(&mut c, "e1", "a.txt", "hello", "bye");
        assert!(!o.ok && o.text.contains("read it first"), "{}", o.text);

        rd(&mut c, "a.txt");
        let f = fs::OpenOptions::new().write(true).open(c.root.join("a.txt")).unwrap();
        f.set_modified(SystemTime::now() + Duration::from_secs(3600)).unwrap();
        drop(f);
        let o = ed(&mut c, "e2", "a.txt", "hello", "bye");
        assert!(!o.ok && o.text.contains("changed since read"), "{}", o.text);
        assert_eq!(gets(&c, "a.txt"), "hello\n");
    }

    #[test]
    fn edit_refuses_non_utf8() {
        let mut c = ctx("edit-utf8");
        put(&c, "l1.txt", [0x66u8, 0xE9, 0x0A]);
        rd(&mut c, "l1.txt");
        let o = ed(&mut c, "e1", "l1.txt", "f", "g");
        assert!(!o.ok && o.text.contains("not valid UTF-8"), "{}", o.text);
        assert_eq!(get(&c, "l1.txt"), [0x66, 0xE9, 0x0A]);
    }

    #[test]
    fn edit_backs_up_before_change() {
        let mut c = ctx("edit-backup");
        put(&c, "a.txt", "one\ntwo\n");
        rd(&mut c, "a.txt");
        let o = ed(&mut c, "call-7", "a.txt", "two", "2");
        assert!(o.ok, "{}", o.text);
        assert_eq!(gets(&c, "a.txt"), "one\n2\n");
        assert_eq!(fs::read_to_string(c.backups_dir.join("call-7.before")).unwrap(), "one\ntwo\n");
        // reads refreshed: a second edit needs no re-read
        assert!(ed(&mut c, "call-8", "a.txt", "one", "1").ok);
    }

    // Extra coverage for the Decisions (not among the brief's ten).
    #[test]
    fn args_missing_or_mistyped_fail_cleanly() {
        let mut c = ctx("args");
        assert_eq!(read(&mut c, &json!({})).text, "missing argument `path`");
        assert_eq!(read(&mut c, &json!({ "path": 5 })).text, "missing argument `path`");
        assert!(!read(&mut c, &json!({ "path": "f", "offset": "2" })).ok);
        assert_eq!(write(&mut c, "w", &json!({ "path": "a" })).text, "missing argument `content`");
        assert_eq!(edit(&mut c, "e", &json!({ "path": "a", "old": "x" })).text, "missing argument `new`");
        assert!(!read(&mut c, &Value::Null).ok);
    }

    #[test]
    fn edit_rejects_empty_old_and_ambiguous_fuzzy_match() {
        let mut c = ctx("edit-edge");
        put(&c, "a.txt", "x\n");
        rd(&mut c, "a.txt");
        let o = ed(&mut c, "e1", "a.txt", "", "y");
        assert!(!o.ok && o.text.contains("must not be empty"), "{}", o.text);

        put(&c, "b.txt", " a\n b\n a\n b\n");
        rd(&mut c, "b.txt");
        let o = ed(&mut c, "e2", "b.txt", "a\nb", "z");
        assert!(!o.ok && o.text.contains("lines 1, 3"), "{}", o.text);
        assert_eq!(gets(&c, "b.txt"), " a\n b\n a\n b\n");
    }

    // ---- shell (Task 5) ----

    #[cfg(windows)]
    #[test]
    fn never_picks_wsl_launcher() {
        let all = |_: &Path| true;
        assert_eq!(
            resolve_shell_from(true, Some(Path::new(r"C:\Program Files\Git\cmd\git.exe")), &[], &all),
            Shell::Bash(PathBuf::from(r"C:\Program Files\Git\bin\bash.exe"))
        );
        let only_wsl = |p: &Path| p.ends_with(r"System32\bash.exe");
        assert_eq!(resolve_shell_from(true, None, &[PathBuf::from(r"C:\Program Files\Git\bin\bash.exe")], &only_wsl), Shell::PowerShell);
        // the WSL launcher is skipped even when it is the only candidate that exists, in any case or slash style
        let wsl = |p: &Path| p.to_string_lossy().to_lowercase().contains("system32");
        assert_eq!(resolve_shell_from(true, None, &[PathBuf::from(r"C:\WINDOWS/System32/bash.exe")], &wsl), Shell::PowerShell);
    }

    #[cfg(windows)]
    #[test]
    fn windows_tries_git_then_extras_in_order() {
        let git = Path::new(r"C:\Git\cmd\git.exe");
        let extra = [PathBuf::from(r"C:\A\bash.exe"), PathBuf::from(r"C:\B\bash.exe")];
        let has = |want: &'static [&'static str]| move |p: &Path| want.iter().any(|w| p == Path::new(w));
        let all = has(&[r"C:\Git\bin\bash.exe", r"C:\A\bash.exe", r"C:\B\bash.exe"]);
        assert_eq!(resolve_shell_from(true, Some(git), &extra, &all), Shell::Bash(PathBuf::from(r"C:\Git\bin\bash.exe")));
        let no_git = has(&[r"C:\A\bash.exe", r"C:\B\bash.exe"]);
        assert_eq!(resolve_shell_from(true, Some(git), &extra, &no_git), Shell::Bash(PathBuf::from(r"C:\A\bash.exe")));
        let last = has(&[r"C:\B\bash.exe"]);
        assert_eq!(resolve_shell_from(true, Some(git), &extra, &last), Shell::Bash(PathBuf::from(r"C:\B\bash.exe")));
        assert_eq!(resolve_shell_from(true, None, &[], &|_| true), Shell::PowerShell);
    }

    #[test]
    fn non_windows_prefers_bash_then_sh() {
        assert_eq!(resolve_shell_from(false, None, &[], &|p| p == Path::new("/bin/bash")), Shell::Bash(PathBuf::from("bash")));
        assert_eq!(resolve_shell_from(false, None, &[], &|p| p == Path::new("/usr/bin/bash")), Shell::Bash(PathBuf::from("bash")));
        assert_eq!(resolve_shell_from(false, None, &[], &|_| false), Shell::Sh);
    }

    #[test]
    fn bash_timeout_defaults_and_clamps() {
        assert_eq!(bash_timeout(&json!({})), Duration::from_secs(120));
        assert_eq!(bash_timeout(&json!({"timeout": 9999})), Duration::from_secs(600));
        assert_eq!(bash_timeout(&json!({"timeout": 5})), Duration::from_secs(5));
        assert_eq!(bash_timeout(&json!({"timeout": 0})), Duration::from_secs(1));
        assert_eq!(bash_timeout(&json!({"timeout": -3})), Duration::from_secs(1));
        assert_eq!(bash_timeout(&json!({"timeout": 2.5})), Duration::from_secs(2));
        assert_eq!(bash_timeout(&json!({"timeout": "soon"})), Duration::from_secs(120));
    }

    // The tests below return early unless resolve_shell() is Shell::Bash.
    fn git_bash() -> Option<Shell> {
        Some(resolve_shell()).filter(|s| matches!(s, Shell::Bash(_)))
    }
    fn run_in(sh: &Shell, dir: &Path, cmd: &str, secs: u64) -> Exec {
        exec_shell(sh, cmd, dir, Duration::from_secs(secs), &AtomicBool::new(false))
    }
    fn run(sh: &Shell, cmd: &str, secs: u64) -> Exec {
        run_in(sh, &test_dir("exec"), cmd, secs)
    }

    #[test]
    fn exec_captures_output_and_exit() {
        let Some(sh) = git_bash() else { return };
        let e = run(&sh, "echo hi; echo err >&2; exit 3", 10);
        assert!(e.output.contains("hi") && e.output.contains("err"), "{:?}", e.output);
        assert_eq!(e.exit_code, Some(3));
        assert!(!e.timed_out && !e.cancelled);
    }

    #[test]
    fn exec_times_out_and_kills_tree() {
        let Some(sh) = git_bash() else { return };
        let e = run(&sh, "echo before; sleep 30", 1);
        assert!(e.timed_out && !e.cancelled, "{e:?}");
        assert_eq!(e.exit_code, None);
        assert!(e.output.contains("before"), "partial output is kept: {:?}", e.output);
        assert!(e.duration_ms >= 1000 && e.duration_ms < 5000, "{}", e.duration_ms);
    }

    #[test]
    fn exec_cancel() {
        let Some(sh) = git_bash() else { return };
        let dir = test_dir("exec-cancel");
        let cancel = AtomicBool::new(false);
        let e = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                cancel.store(true, Ordering::SeqCst);
            });
            exec_shell(&sh, "sleep 30", &dir, Duration::from_secs(20), &cancel)
        });
        assert!(e.cancelled && !e.timed_out, "{e:?}");
        assert_eq!(e.exit_code, None);
        assert!(e.duration_ms < 5000, "{}", e.duration_ms);
    }

    #[test]
    fn stdin_is_closed() {
        let Some(sh) = git_bash() else { return };
        let e = run(&sh, "cat", 10);
        assert_eq!(e.exit_code, Some(0), "{e:?}");
        assert!(!e.timed_out && e.duration_ms < 5000, "{e:?}");
    }

    #[test]
    fn background_process_does_not_hang() {
        let Some(sh) = git_bash() else { return };
        let e = run(&sh, "sleep 20 & echo started", 30);
        assert!(e.duration_ms < 5000, "{}", e.duration_ms);
        assert!(e.output.contains("started"), "{:?}", e.output);
        assert_eq!(e.exit_code, Some(0));
    }

    #[test]
    fn runs_in_root_with_spaces() {
        let Some(sh) = git_bash() else { return };
        let dir = test_dir("exec-space").join("with space");
        fs::create_dir_all(&dir).unwrap();
        let e = run_in(&sh, &dir, "pwd", 10);
        assert!(e.output.contains("with space"), "{:?}", e.output);
        assert_eq!(e.exit_code, Some(0));
    }

    #[test]
    fn exec_passes_quotes_and_newlines_intact() {
        let Some(sh) = git_bash() else { return };
        let e = run(&sh, "echo \"a b\" 'c  d'\nx=1\necho \"x=$x\" back\\\\slash", 10);
        assert_eq!(e.output.replace("\r\n", "\n"), "a b c  d\nx=1 back\\slash\n", "{:?}", e.output);
    }

    /// On Windows the command crosses a command line: backslash runs and quotes must reach bash unchanged.
    #[test]
    fn exec_passes_backslashes_and_quotes_through_unchanged() {
        let Some(sh) = git_bash() else { return };
        let dir = test_dir("exec-bs");
        for text in [r"a\\b", r"a\\\b", r"\\\\", r"trail\\", r"trail\", r#"x\"y"#, r#"x\\"y"#, r#"q"q"#, r"it's `x` $HOME", "tab\\t nl\\n", "two\nlines  gap\n\n"] {
            let e = run_in(&sh, &dir, &format!("cat <<'XOLOTL_EOF'\n{text}\nXOLOTL_EOF"), 10);
            assert_eq!(e.output, format!("{text}\n"), "text {text:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn powershell_fallback_runs() {
        let e = run(&Shell::PowerShell, "Write-Output 'a b'; exit 4", 30);
        assert!(e.output.contains("a b"), "{:?}", e.output);
        assert_eq!(e.exit_code, Some(4));
    }

    #[test]
    fn exec_keeps_only_the_first_capture_limit_of_runaway_output() {
        let Some(sh) = git_bash() else { return };
        let e = run(&sh, &format!("head -c {} /dev/zero | tr '\\0' x", MAX_CAPTURE + 5_000_000), 30);
        assert_eq!(e.exit_code, Some(0), "{:?}", clip(&e.output, 200));
        assert!(e.output.starts_with("xxxx") && e.output.len() > MAX_CAPTURE, "{}", e.output.len());
        assert!(e.output.ends_with("discarded]"), "{:?}", clip_tail(&e.output, 100));
        assert!(e.output.len() < MAX_CAPTURE + 200, "{}", e.output.len());
    }

    #[test]
    fn bash_tool_footer() {
        let Some(shell) = git_bash() else { return };
        let mut c = ctx("bash-tool");
        c.shell = shell;
        let o = bash(&mut c, &json!({ "command": "echo hi" }), &AtomicBool::new(false));
        assert!(o.ok && o.exit_code == Some(0) && !o.cancelled, "{o:?}");
        assert!(o.text.starts_with("hi"), "{:?}", o.text);
        let footer = o.text.rsplit('\n').next().unwrap();
        let secs = footer.strip_prefix("[exit 0 · ").and_then(|f| f.strip_suffix("s]")).unwrap_or_else(|| panic!("{footer:?}"));
        assert!(secs.parse::<f64>().is_ok() && secs.split('.').nth(1).map(str::len) == Some(1), "{secs:?}");

        let o = bash(&mut c, &json!({ "command": "exit 2" }), &AtomicBool::new(false));
        assert!(!o.ok && o.exit_code == Some(2), "{o:?}");
        assert!(o.text.contains("[exit 2 · "), "{:?}", o.text);

        let o = bash(&mut c, &json!({ "command": "sleep 5", "timeout": 1 }), &AtomicBool::new(false));
        assert!(!o.ok && !o.cancelled && o.exit_code.is_none(), "{o:?}");
        assert!(o.text.contains("[timed out after 1s · partial output above]"), "{:?}", o.text);
    }

    #[test]
    fn bash_tool_cancel_and_bad_args() {
        let Some(shell) = git_bash() else { return };
        let mut c = ctx("bash-tool-cancel");
        c.shell = shell;
        let o = bash(&mut c, &json!({ "command": "sleep 30" }), &AtomicBool::new(true));
        assert!(!o.ok && o.cancelled, "{o:?}");
        assert!(o.text.ends_with("\n[cancelled by user]"), "{:?}", o.text);

        assert_eq!(bash(&mut c, &json!({}), &AtomicBool::new(false)).text, "missing argument `command`");
        assert!(!bash(&mut c, &json!({ "command": 7 }), &AtomicBool::new(false)).ok);
    }
}

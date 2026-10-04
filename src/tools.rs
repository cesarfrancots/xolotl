//! File tools `read`, `write`, `edit` (spec §7, §7.1) and the char-boundary-safe `clip` helpers.

use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAX_READ_LINES: usize = 2000;
const MAX_LINE_BYTES: usize = 2000;
const BINARY_SNIFF_BYTES: usize = 8192;

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
}

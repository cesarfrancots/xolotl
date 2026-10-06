use crate::llm::{ToolCall, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub seq: u64,
    pub t_ms: u64,
    #[serde(flatten)]
    pub kind: Kind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Kind {
    SessionStart { cwd: String, model: String, mode: String, policy_hash: String, system: String },
    User { text: String },
    Assistant {
        text: String,
        reasoning: String,
        tool_calls: Vec<ToolCall>,
        usage: Usage,
        finish_reason: String,
        latency_ms: u64,
        /// The `reasoning_effort` the request carried; absent: none was sent (the API default, `max`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effort: Option<String>,
        /// Milliseconds of reasoning before the first content or tool-call token (the whole call if none came).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        think_ms: Option<u64>,
    },
    ToolResult {
        call_id: String,
        name: String,
        ok: bool,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        artifact: Option<String>,
        bytes: u64,
        duration_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
    },
    /// decision: allow|deny|always; by: user|rule|mode
    Permission { call_id: String, decision: String, by: String },
    Compaction {
        stage: u8,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        upto_seq: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_seq: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        to_seq: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary_path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_before: Option<u64>,
    },
    /// reason: completion_check|repeat|empty|time|think_cap
    Nudge { reason: String, text: String },
    /// during: stream|tool
    Interrupt {
        during: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
    },
    Error { kind: String, message: String, retry: bool },
    SessionEnd { status: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub id: String,
    pub cwd: String,
    pub git_head: Option<String>,
    pub model: String,
    pub mode: String,
    pub policy_hash: String,
    pub started_ms: u64,
    pub ended_ms: Option<u64>,
    pub status: String,
}

pub struct Session {
    pub dir: PathBuf,
    pub events: Vec<Event>,
    pub meta: Meta,
    file: File,
}

fn bad(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

/// Every char that isn't ASCII alphanumeric, `-`, `_` or `.` becomes `-`.
fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect()
}

impl Session {
    /// `root` = `~/.xolotl`. Fails with `AlreadyExists` rather than reusing an existing session directory.
    pub fn create(root: &Path, meta: Meta) -> io::Result<Session> {
        let sessions = root.join("projects").join(project_slug(Path::new(&meta.cwd))).join("sessions");
        fs::create_dir_all(&sessions)?;
        let dir = sessions.join(&meta.id);
        fs::create_dir(&dir)?;
        fs::create_dir(dir.join("artifacts"))?;
        fs::create_dir(dir.join("summaries"))?;
        let file = OpenOptions::new().create(true).append(true).open(dir.join("events.jsonl"))?;
        let s = Session { dir, events: Vec::new(), meta, file };
        s.write_meta()?;
        Ok(s)
    }

    /// Loads a session for resume. A partial last line (crash mid-write) is truncated away first so `seq` == line number still holds.
    pub fn open(dir: &Path) -> io::Result<Session> {
        let meta = serde_json::from_slice(&fs::read(dir.join("meta.json"))?)?;
        let path = dir.join("events.jsonl");
        let mut raw = fs::read(&path)?;
        if raw.last().is_some_and(|b| *b != b'\n') {
            let keep = raw.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
            OpenOptions::new().write(true).open(&path)?.set_len(keep as u64)?;
            raw.truncate(keep);
        }
        let mut events = Vec::new();
        for (i, line) in std::str::from_utf8(&raw).map_err(bad)?.lines().enumerate() {
            let n = i as u64 + 1;
            let e: Event = serde_json::from_str(line).map_err(|e| bad(format!("events.jsonl line {n}: {e}")))?;
            if e.seq != n {
                return Err(bad(format!("events.jsonl line {n} has seq {}", e.seq)));
            }
            events.push(e);
        }
        let file = OpenOptions::new().append(true).open(&path)?;
        Ok(Session { dir: dir.to_path_buf(), events, meta, file })
    }

    pub fn append(&mut self, kind: Kind) -> io::Result<Event> {
        let event = Event { seq: self.events.len() as u64 + 1, t_ms: now_ms(), kind };
        let mut line = serde_json::to_string(&event)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.flush()?;
        self.events.push(event.clone());
        Ok(event)
    }

    /// Gives every assistant tool call without a result a synthetic failed result; returns how many were added.
    pub fn repair_dangling(&mut self) -> io::Result<usize> {
        // ponytail: matches on call id across the whole history; fine while the API keeps ids unique per session.
        let missing: Vec<(String, String)> = {
            let answered: HashSet<&str> = self
                .events
                .iter()
                .filter_map(|e| match &e.kind {
                    Kind::ToolResult { call_id, .. } => Some(call_id.as_str()),
                    _ => None,
                })
                .collect();
            self.events
                .iter()
                .filter_map(|e| match &e.kind {
                    Kind::Assistant { tool_calls, .. } => Some(tool_calls),
                    _ => None,
                })
                .flatten()
                .filter(|c| !answered.contains(c.id.as_str()))
                .map(|c| (c.id.clone(), c.name.clone()))
                .collect()
        };
        let n = missing.len();
        for (call_id, name) in missing {
            let output = "interrupted (session resumed)".to_string();
            let bytes = output.len() as u64;
            self.append(Kind::ToolResult { call_id, name, ok: false, output, artifact: None, bytes, duration_ms: 0, exit_code: None })?;
        }
        Ok(n)
    }

    #[allow(dead_code)] // only the tests use it in v1
    pub fn events_path(&self) -> PathBuf {
        self.dir.join("events.jsonl")
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.dir.join("artifacts")
    }

    /// `name` usually derives from a model-supplied call id, so it is sanitized to stay a single file name inside `artifacts/`.
    pub fn write_artifact(&self, name: &str, data: &[u8]) -> io::Result<PathBuf> {
        let path = self.artifacts_dir().join(sanitize(name));
        fs::write(&path, data)?;
        Ok(path)
    }

    /// Writes `summaries/<n>.md` where n = existing summaries + 1.
    pub fn write_summary(&self, text: &str) -> io::Result<PathBuf> {
        let dir = self.dir.join("summaries");
        let path = dir.join(format!("{}.md", fs::read_dir(&dir)?.count() + 1));
        fs::write(&path, text)?;
        Ok(path)
    }

    pub fn write_env(&self, text: &str) -> io::Result<()> {
        fs::write(self.dir.join("env.txt"), text)
    }

    pub fn finish(&mut self, status: &str) -> io::Result<()> {
        self.append(Kind::SessionEnd { status: status.into() })?;
        self.meta.ended_ms = Some(now_ms());
        self.meta.status = status.into();
        self.write_meta()
    }

    /// Marks a resumed session as running again, so a crash after resume doesn't leave it reading as finished.
    pub fn reopen(&mut self) -> io::Result<()> {
        self.meta.ended_ms = None;
        self.meta.status = "running".into();
        self.write_meta()
    }

    // ponytail: plain overwrite, not temp-file + rename; a crash mid-write could leave meta.json truncated. events.jsonl is the source of truth.
    fn write_meta(&self) -> io::Result<()> {
        fs::write(self.dir.join("meta.json"), serde_json::to_string_pretty(&self.meta)?)
    }
}

pub fn project_slug(cwd: &Path) -> String {
    sanitize(&cwd.to_string_lossy())
}

/// UTC `YYYYMMDD-HHMMSS-<rand as 4 hex>`; sorts chronologically.
pub fn session_id(now_ms: u64, rand: u16) -> String {
    let secs = now_ms / 1000;
    let (y, m, d) = civil_from_days(secs / 86_400);
    let r = secs % 86_400;
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}-{rand:04x}", r / 3600, r % 3600 / 60, r % 60)
}

/// Howard Hinnant's civil_from_days: days since 1970-01-01 -> (year, month, day). Unsigned, so dates before 1970 are out of range.
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (m <= 2) as u64, m, d)
}

/// Session directories of `cwd`'s project, newest first.
pub fn list_sessions(root: &Path, cwd: &Path) -> Vec<PathBuf> {
    let dir = root.join("projects").join(project_slug(cwd)).join("sessions");
    let mut v: Vec<PathBuf> = fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    v.sort(); // ids are UTC timestamps, so name order is time order
    v.reverse();
    v
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}

/// Fresh, unique directory under `temp_dir()/xolotl-tests/<tag>-<nanos>-<n>` (the counter keeps parallel tests apart).
#[cfg(test)]
pub fn test_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join("xolotl-tests").join(format!("{tag}-{nanos}-{}", N.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn meta(id: &str) -> Meta {
        Meta {
            id: id.into(),
            cwd: "/work/my proj".into(),
            git_head: None,
            model: "glm-5.3-flash".into(),
            mode: "writes".into(),
            policy_hash: "abc".into(),
            started_ms: 1,
            ended_ms: None,
            status: "running".into(),
        }
    }

    fn user(t: &str) -> Kind {
        Kind::User { text: t.into() }
    }

    fn parse_lines(path: &Path) -> Vec<Event> {
        let raw = fs::read_to_string(path).unwrap();
        assert!(raw.is_empty() || raw.ends_with('\n'));
        assert!(!raw.contains('\r'));
        raw.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    #[test]
    fn slug_and_id() {
        assert_eq!(project_slug(Path::new(r"D:\Xolotl Harness")), "D--Xolotl-Harness");
        assert_eq!(project_slug(Path::new("/home/me/my proj")), "-home-me-my-proj");
        assert_eq!(session_id(0, 0xa3f2), "19700101-000000-a3f2");
        assert_eq!(session_id(1_759_569_300_123, 0x00ff), "20251004-091500-00ff");
        assert_eq!(session_id(1_790_000_000_000, 1), "20260921-141320-0001");
    }

    #[test]
    fn fnv_known_values() {
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn seq_equals_line_number() {
        let root = test_dir("seq");
        let mut s = Session::create(&root, meta("20250101-000000-0001")).unwrap();
        assert_eq!(s.append(user("a\nb")).unwrap().seq, 1);
        assert_eq!(s.append(user("c")).unwrap().seq, 2);
        let raw = fs::read_to_string(s.events_path()).unwrap();
        assert_eq!(raw.lines().count(), 2);
        let evs = parse_lines(&s.events_path());
        for (i, e) in evs.iter().enumerate() {
            assert_eq!(e.seq, i as u64 + 1);
        }
        assert!(raw.lines().next().unwrap().contains("\"type\":\"user\""));
        assert_eq!(evs[0].kind, user("a\nb"));
        assert_eq!(evs, s.events);
    }

    #[test]
    fn kinds_round_trip() {
        let call = ToolCall { id: "c1".into(), name: "bash".into(), args: r#"{"command":"ls"}"#.into() };
        let kinds = vec![
            Kind::SessionStart { cwd: "/p".into(), model: "m".into(), mode: "writes".into(), policy_hash: "h".into(), system: "sys".into() },
            user("hi"),
            Kind::Assistant {
                text: "t".into(),
                reasoning: "r".into(),
                tool_calls: vec![call],
                usage: Usage { prompt: 10, completion: 2, cached: 3 },
                finish_reason: "tool_calls".into(),
                latency_ms: 7,
                effort: Some("medium".into()),
                think_ms: Some(5),
            },
            Kind::ToolResult { call_id: "c1".into(), name: "bash".into(), ok: true, output: "o".into(), artifact: Some("artifacts/c1.txt".into()), bytes: 1, duration_ms: 2, exit_code: Some(-1) },
            Kind::ToolResult { call_id: "c2".into(), name: "read".into(), ok: false, output: "o".into(), artifact: None, bytes: 1, duration_ms: 2, exit_code: None },
            Kind::Permission { call_id: "c1".into(), decision: "allow".into(), by: "user".into() },
            Kind::Compaction { stage: 1, upto_seq: Some(9), from_seq: None, to_seq: None, summary_path: None, summary: None, tokens_before: None },
            Kind::Compaction {
                stage: 2,
                upto_seq: None,
                from_seq: Some(1),
                to_seq: Some(9),
                summary_path: Some("summaries/1.md".into()),
                summary: Some("s".into()),
                tokens_before: Some(150_000),
            },
            Kind::Nudge { reason: "empty".into(), text: "go on".into() },
            Kind::Interrupt { during: "tool".into(), call_id: Some("c1".into()) },
            Kind::Interrupt { during: "stream".into(), call_id: None },
            Kind::Error { kind: "http".into(), message: "boom".into(), retry: true },
            Kind::SessionEnd { status: "done".into() },
        ];
        for (i, kind) in kinds.into_iter().enumerate() {
            let e = Event { seq: i as u64 + 1, t_ms: 42, kind };
            let line = serde_json::to_string(&e).unwrap();
            assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), e, "{line}");
            assert!(!line.contains("null"), "None must be omitted: {line}");
        }
    }

    #[test]
    fn open_truncates_partial_last_line() {
        let root = test_dir("partial");
        let mut s = Session::create(&root, meta("20250101-000000-0001")).unwrap();
        s.append(user("a")).unwrap();
        s.append(user("b")).unwrap();
        let (dir, path) = (s.dir.clone(), s.events_path());
        drop(s);
        fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(br#"{"seq":3,"t_ms":1,"ty"#).unwrap();

        let mut s = Session::open(&dir).unwrap();
        assert_eq!(s.events.len(), 2);
        assert_eq!(s.append(user("c")).unwrap().seq, 3);
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 3);
        let evs = parse_lines(&path);
        assert_eq!(evs.iter().map(|e| e.seq).collect::<Vec<_>>(), [1, 2, 3]);

        // a partial line with no complete line before it truncates to empty
        let mut s2 = Session::create(&root, meta("20250101-000000-0002")).unwrap();
        let (dir2, path2) = (s2.dir.clone(), s2.events_path());
        s2.file.write_all(br#"{"seq":1,"t_"#).unwrap();
        drop(s2);
        let mut s2 = Session::open(&dir2).unwrap();
        assert!(s2.events.is_empty());
        assert_eq!(fs::metadata(&path2).unwrap().len(), 0);
        assert_eq!(s2.append(user("x")).unwrap().seq, 1);
    }

    #[test]
    fn assistant_trace_fields_are_additive() {
        let a = |effort: Option<&str>, think_ms: Option<u64>| Kind::Assistant {
            text: "t".into(),
            reasoning: String::new(),
            tool_calls: vec![],
            usage: Usage::default(),
            finish_reason: "stop".into(),
            latency_ms: 1,
            effort: effort.map(Into::into),
            think_ms,
        };
        // an event without them serializes exactly as it did before, and an old line still loads
        let bare = serde_json::to_string(&a(None, None)).unwrap();
        assert!(!bare.contains("effort") && !bare.contains("think_ms"), "{bare}");
        assert_eq!(serde_json::from_str::<Kind>(&bare).unwrap(), a(None, None));
        let full = serde_json::to_string(&a(Some("low"), Some(1500))).unwrap();
        assert!(full.contains(r#""effort":"low""#) && full.contains(r#""think_ms":1500"#), "{full}");
        assert_eq!(serde_json::from_str::<Kind>(&full).unwrap(), a(Some("low"), Some(1500)));
    }

    #[test]
    fn repair_dangling_tool_calls() {
        let root = test_dir("dangling");
        let mut s = Session::create(&root, meta("20250101-000000-0001")).unwrap();
        let call = |id: &str, name: &str| ToolCall { id: id.into(), name: name.into(), args: "{}".into() };
        s.append(Kind::Assistant {
            text: String::new(),
            reasoning: String::new(),
            tool_calls: vec![call("c1", "bash"), call("c2", "read")],
            usage: Usage::default(),
            finish_reason: "tool_calls".into(),
            latency_ms: 5,
            effort: None,
            think_ms: None,
        })
        .unwrap();
        s.append(Kind::ToolResult { call_id: "c1".into(), name: "bash".into(), ok: true, output: "x".into(), artifact: None, bytes: 1, duration_ms: 1, exit_code: Some(0) }).unwrap();
        assert_eq!(s.repair_dangling().unwrap(), 1);
        assert_eq!(
            s.events.last().unwrap().kind,
            Kind::ToolResult {
                call_id: "c2".into(),
                name: "read".into(),
                ok: false,
                output: "interrupted (session resumed)".into(),
                artifact: None,
                bytes: 29,
                duration_ms: 0,
                exit_code: None,
            }
        );
        assert_eq!(s.events.len(), 3);
        assert_eq!(s.repair_dangling().unwrap(), 0);
        assert_eq!(parse_lines(&s.events_path()).len(), 3);
    }

    #[test]
    fn layout_meta_and_files() {
        let root = test_dir("layout");
        let m = meta("20250101-000000-0001");
        let mut s = Session::create(&root, m.clone()).unwrap();
        assert_eq!(s.dir, root.join("projects").join("-work-my-proj").join("sessions").join(&m.id));
        assert!(s.artifacts_dir().is_dir());
        assert!(s.dir.join("summaries").is_dir());
        let read_meta = |dir: &Path| serde_json::from_str::<Meta>(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(read_meta(&s.dir), m);

        assert_eq!(s.write_summary("one").unwrap(), s.dir.join("summaries").join("1.md"));
        assert_eq!(s.write_summary("two").unwrap(), s.dir.join("summaries").join("2.md"));
        assert_eq!(fs::read_to_string(s.dir.join("summaries").join("2.md")).unwrap(), "two");
        s.write_env("os=windows").unwrap();
        assert_eq!(fs::read_to_string(s.dir.join("env.txt")).unwrap(), "os=windows");

        let p = s.write_artifact("c1.txt", b"hi").unwrap();
        assert_eq!(p, s.artifacts_dir().join("c1.txt"));
        assert_eq!(fs::read(&p).unwrap(), b"hi");
        // call ids come from the model: names must not escape artifacts/
        let p = s.write_artifact(r"..\..\evil:1", b"x").unwrap();
        assert_eq!(p.parent().unwrap(), s.artifacts_dir());
        assert!(!s.dir.join("evil:1").exists());

        s.finish("done").unwrap();
        assert_eq!(s.events.last().unwrap().kind, Kind::SessionEnd { status: "done".into() });
        let on_disk = read_meta(&s.dir);
        assert_eq!(on_disk.status, "done");
        assert!(on_disk.ended_ms.is_some());
        assert_eq!(on_disk, s.meta);

        let reopened = Session::open(&s.dir).unwrap();
        assert_eq!(reopened.meta, s.meta);
        assert_eq!(reopened.events, s.events);
    }

    #[test]
    fn reopen_marks_a_finished_session_running() {
        let root = test_dir("reopen");
        let mut s = Session::create(&root, meta("20250101-000000-0001")).unwrap();
        s.finish("done").unwrap();
        let mut s = Session::open(&s.dir).unwrap();
        s.reopen().unwrap();
        assert_eq!((s.meta.status.as_str(), s.meta.ended_ms), ("running", None));
        assert_eq!(Session::open(&s.dir).unwrap().meta, s.meta);
    }

    #[test]
    fn list_sessions_newest_first() {
        let root = test_dir("list");
        let cwd = Path::new("/work/my proj");
        for id in ["20250101-000000-0001", "20260101-000000-0001"] {
            Session::create(&root, meta(id)).unwrap();
        }
        let mut other = meta("20270101-000000-0001");
        other.cwd = "/elsewhere".into();
        Session::create(&root, other).unwrap();

        let names: Vec<String> = list_sessions(&root, cwd).iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["20260101-000000-0001", "20250101-000000-0001"]);
        assert!(list_sessions(&root, Path::new("/nowhere")).is_empty());
    }
}

//! The program (spec §6, §13): flags and config, startup, the headless run (`-p`) and the interactive REPL.

mod agent;
mod llm;
mod perms;
mod policy;
mod store;
mod tools;
mod ui;

use agent::{Agent, Outcome};
use crossterm::event::{self, KeyEvent, KeyEventKind};
use crossterm::style::Stylize;
use llm::{ZaiClient, ZaiConfig};
use perms::{Mode, Perms};
use reedline::{
    ColumnarMenu, DefaultCompleter, DefaultPrompt, DefaultPromptSegment, EditCommand, Emacs, FileBackedHistory, KeyCode, KeyModifiers,
    MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal, default_emacs_keybindings,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::{env, fs};
use store::{Event, Kind, Meta, Session};
use tools::{Shell, ToolCtx};
use ui::{Headless, Status, Terminal, Ui, status_line};

/// Set by Ctrl+C: cancels the running stream or tool. While the line editor reads input, the terminal is in raw
/// mode and Ctrl+C arrives as a key instead (it clears the line).
static CANCEL: AtomicBool = AtomicBool::new(false);

const USAGE: &str =
    "usage: xolotl [-p <task>] [--mode ask|writes|yolo] [--cwd <dir>] [--resume [id]] [--budget <tokens>] [--allow <bash-prefix>]... [--model <id>] [--version]";
/// The `reasoning_effort` values the API documents. It silently accepts any other value (docs/glm-api-notes.md #5).
const EFFORTS: [&str; 7] = ["max", "xhigh", "high", "medium", "low", "minimal", "none"];
const SLASH_COMMANDS: [&str; 8] = ["/help", "/mode", "/compact", "/clear", "/resume", "/think", "/status", "/exit"];
const HELP: &str = "\
/help                     this list
/mode [ask|writes|yolo]   show or set the permission mode
/compact [focus]          summarize the conversation so far
/clear                    end this session and start a new one
/resume [id]              continue an earlier session (lists recent ones without an id)
/think                    show the model's reasoning in full, or as one line
/status                   context use, tool calls, time and cache hits
/exit                     quit (also Ctrl+D)
Alt+Enter inserts a newline. Ctrl+C clears the line, or cancels a running request.";
const UNWRITABLE: &str = "this session can no longer be written — /clear starts a new one (or /resume another), or restart xolotl";
const HISTORY_SIZE: usize = 1000;
/// Sessions `/resume` lists to pick from.
const RESUME_LIST: usize = 10;

#[derive(Debug, Default, PartialEq)]
struct Args {
    prompt: Option<String>,
    mode: Option<Mode>,
    cwd: Option<PathBuf>,
    /// `Some(None)`: `--resume` without an id (the newest session).
    resume: Option<Option<String>>,
    budget: Option<usize>,
    allow: Vec<String>,
    model: Option<String>,
    version: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = args.iter().peekable();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-p" => a.prompt = Some(value()?),
            "--mode" => a.mode = Some(value()?.parse()?),
            "--cwd" => a.cwd = Some(value()?.into()),
            "--budget" => {
                let v = value()?;
                a.budget = Some(v.parse().map_err(|_| format!("--budget needs a number of tokens, not `{v}`"))?);
            }
            "--allow" => a.allow.push(value()?),
            "--model" => a.model = Some(value()?),
            "--version" => a.version = true,
            "--resume" => a.resume = Some(it.next_if(|t| !t.starts_with('-')).cloned()),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(a)
}

/// `~/.xolotl/config.json`; every field is optional.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Config {
    base_url: String,
    model: String,
    /// Name of the environment variable that holds the API key; the key itself is never in the config.
    api_key_env: String,
    context_budget: usize,
    mode: String,
    temperature: f64,
    top_p: f64,
    reasoning_effort: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            base_url: "https://api.z.ai/api/coding/paas/v4".into(), // Coding Plan endpoint (docs/glm-api-notes.md #1)
            model: "glm-5.3-flash".into(),
            api_key_env: "ZAI_API_KEY".into(),
            context_budget: 200_000,
            mode: "writes".into(),
            temperature: 1.0,
            top_p: 0.95,
            reasoning_effort: Some("max".into()),
        }
    }
}

/// A missing file means the defaults.
fn load_config(path: &Path) -> Result<Config, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Config::default()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    // Windows editors and PowerShell's `-Encoding utf8` start the file with a BOM, which JSON doesn't allow
    serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| format!("{}: {e}", path.display()))
}

/// The config with the flags laid over it, validated, and the permission mode to start in.
fn apply(mut cfg: Config, args: &Args) -> Result<(Config, Mode), String> {
    let mode = match args.mode {
        Some(m) => m,
        None => cfg.mode.parse().map_err(|e| format!("config.json mode: {e}"))?,
    };
    if let Some(m) = &args.model {
        cfg.model = m.clone();
    }
    if let Some(b) = args.budget {
        cfg.context_budget = b;
    }
    if cfg.context_budget == 0 {
        return Err("the context budget must be above 0".into());
    }
    if let Some(e) = cfg.reasoning_effort.as_deref().filter(|e| !EFFORTS.contains(e)) {
        return Err(format!("config.json reasoning_effort: unknown value `{e}` (expected one of {})", EFFORTS.join(", ")));
    }
    Ok((cfg, mode))
}

/// The API key from the process environment, else (Windows) from the User scope in the registry: a key set with
/// `SetEnvironmentVariable(.., 'User')` doesn't reach processes whose parent started before that. Never shown or stored.
fn api_key(var: &str) -> Option<String> {
    pick_key(env::var(var).ok(), || user_env(var))
}

/// The first non-blank of the two, trimmed (a trailing newline in the key would fail every request).
fn pick_key(process: Option<String>, user_scope: impl FnOnce() -> Option<String>) -> Option<String> {
    let usable = |k: String| Some(k.trim().to_string()).filter(|k| !k.is_empty());
    process.and_then(usable).or_else(|| user_scope().and_then(usable))
}

fn user_env(var: &str) -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    let out = Command::new("reg").args(["query", r"HKCU\Environment", "/v", var]).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| reg_value(&String::from_utf8_lossy(&out.stdout), var)).flatten()
}

/// The data of value `name` in `reg query` output, whose lines read `    NAME    REG_SZ    data`.
fn reg_value(out: &str, name: &str) -> Option<String> {
    out.lines().find_map(|line| {
        let mut fields = line.trim_start().splitn(3, "    ");
        let (n, ty, data) = (fields.next()?, fields.next()?, fields.next()?);
        (n.eq_ignore_ascii_case(name) && ty.starts_with("REG_")).then(|| data.to_string())
    })
}

/// `--cwd` or the current directory, absolute and without a trailing separator. Not `fs::canonicalize`: on Windows
/// that adds a `\\?\` prefix, under which `.`, `..` and `/` in the model's paths become literal.
fn project_root(cwd: Option<&Path>) -> Result<PathBuf, String> {
    let p = match cwd {
        Some(p) => p.to_path_buf(),
        None => env::current_dir().map_err(|e| format!("cannot read the current directory: {e}"))?,
    };
    let root: PathBuf = std::path::absolute(&p).map_err(|e| format!("{}: {e}", p.display()))?.components().collect();
    if root.is_dir() { Ok(root) } else { Err(format!("{} is not a directory", root.display())) }
}

#[derive(Debug, PartialEq)]
enum Slash {
    Help,
    Mode(Option<String>),
    Compact(Option<String>),
    Clear,
    Resume(Option<String>),
    Think,
    Status,
    Exit,
    Unknown(String),
}

/// `None`: the line is a request, not a slash command.
fn parse_slash(line: &str) -> Option<Slash> {
    let rest = line.trim().strip_prefix('/')?;
    let (cmd, arg) = rest.split_once(char::is_whitespace).map_or((rest, ""), |(c, a)| (c, a.trim()));
    let arg = (!arg.is_empty()).then(|| arg.to_string());
    Some(match cmd {
        "help" => Slash::Help,
        "mode" => Slash::Mode(arg),
        "compact" => Slash::Compact(arg),
        "clear" => Slash::Clear,
        "resume" => Slash::Resume(arg),
        "think" => Slash::Think,
        "status" => Slash::Status,
        "exit" => Slash::Exit,
        _ => Slash::Unknown(cmd.to_string()),
    })
}

/// What startup settles once; `/clear` and `/resume` reuse it.
struct Ctx {
    /// `~/.xolotl`
    root: PathBuf,
    /// The project root, also the process cwd.
    cwd: PathBuf,
    shell: Shell,
    model: String,
    budget: usize,
    /// `--allow` prefixes.
    allow: Vec<String>,
    /// Name of the variable that holds the API key: named in auth errors, kept out of every tool shell.
    api_key_env: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let args = parse_args(&env::args().skip(1).collect::<Vec<_>>()).map_err(|e| format!("{e}\n{USAGE}"))?;
    if args.version {
        println!("xolotl {}", env!("CARGO_PKG_VERSION"));
        return Ok(ExitCode::SUCCESS);
    }
    let root = env::home_dir().ok_or("cannot find the home directory")?.join(".xolotl");
    let (cfg, mode) = apply(load_config(&root.join("config.json"))?, &args)?;
    let api_key = api_key(&cfg.api_key_env).ok_or_else(|| format!("set {} to your Z.ai API key", cfg.api_key_env))?;
    // Every tool shell inherits this process's environment: with the key left in it, `printenv` would put the key
    // into a tool result, events.jsonl and the next request.
    // SAFETY: no other thread exists yet. Only the main thread has run so far (`reg` ran as a child process and
    // has exited); the Ctrl+C handler thread and the model's worker threads start later.
    unsafe { env::remove_var(&cfg.api_key_env) };
    let cwd = project_root(args.cwd.as_deref())?;
    // the permission checks resolve relative paths against the process cwd
    env::set_current_dir(&cwd).map_err(|e| format!("cannot enter {}: {e}", cwd.display()))?;
    ctrlc::set_handler(|| CANCEL.store(true, Ordering::SeqCst)).map_err(|e| format!("cannot handle Ctrl+C: {e}"))?;
    let ctx = Ctx {
        root,
        cwd,
        shell: tools::resolve_shell(),
        model: cfg.model.clone(),
        budget: cfg.context_budget,
        allow: args.allow.clone(),
        api_key_env: cfg.api_key_env.clone(),
    };

    if args.prompt.is_none() {
        // before the session: the environment snapshot can take seconds
        let branch = git(&ctx.cwd, &["branch", "--show-current"]).map(|b| format!(" ({b})")).unwrap_or_default();
        println!("xolotl · {} · {}{branch} · mode: {}", ctx.model, ctx.cwd.display(), mode.as_str());
    }
    // `startup`: index of the first event this process appended before any request (all of a new session's)
    let (session, startup) = match &args.resume {
        Some(id) => {
            let sessions = store::list_sessions(&ctx.root, &ctx.cwd);
            let dir = match id {
                Some(id) => by_id(sessions, id)?,
                None => sessions.into_iter().next().ok_or("no session to resume in this project")?,
            };
            open_session(&dir)?
        }
        None => (new_session(&ctx, mode)?, 0),
    };
    let mut model = ZaiClient::new(ZaiConfig {
        base_url: cfg.base_url,
        model: cfg.model,
        api_key,
        temperature: cfg.temperature,
        top_p: cfg.top_p,
        reasoning_effort: cfg.reasoning_effort,
    });
    Ok(match &args.prompt {
        Some(task) => headless(&ctx, session, startup, &mut model, mode, task),
        None => {
            if args.resume.is_some() {
                println!("resumed {}", session.meta.id);
            }
            repl(&ctx, session, &mut model, mode)
        }
    })
}

/// Trimmed stdout of a successful, non-empty `git` command run in `cwd`.
fn git(cwd: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).current_dir(cwd).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

/// Spec §11.1, §11.2: the environment snapshot, AGENTS.md and the system prompt, then a new session that starts
/// with `session_start`.
fn new_session(ctx: &Ctx, mode: Mode) -> Result<Session, String> {
    CANCEL.store(false, Ordering::SeqCst); // a Ctrl+C from before must not cut the snapshot; a new one may
    let timeout = Duration::from_secs(policy::LIMITS.snapshot_timeout_s);
    // fails silently: on an error or a timeout the snapshot is just shorter
    let scrub = std::slice::from_ref(&ctx.api_key_env);
    let snapshot = tools::exec_shell(&ctx.shell, &policy::env_snapshot_command(&ctx.shell), &ctx.cwd, timeout, &CANCEL, scrub).output;
    let agents_md = fs::read_to_string(ctx.cwd.join("AGENTS.md")).ok();
    let now = store::now_ms();
    let meta = Meta {
        id: store::session_id(now, RandomState::new().hash_one(now) as u16),
        cwd: ctx.cwd.display().to_string(),
        git_head: git(&ctx.cwd, &["rev-parse", "HEAD"]),
        model: ctx.model.clone(),
        mode: mode.as_str().into(),
        policy_hash: format!("{:016x}", store::fnv1a64(include_str!("policy.rs").as_bytes())),
        started_ms: now,
        ended_ms: None,
        status: "running".into(),
    };
    let mut s = Session::create(&ctx.root, meta).map_err(|e| format!("cannot create a session: {e}"))?;
    let system = policy::system_prompt(&snapshot, agents_md.as_deref(), &s.dir);
    let m = &s.meta;
    let start = Kind::SessionStart { cwd: m.cwd.clone(), model: m.model.clone(), mode: m.mode.clone(), policy_hash: m.policy_hash.clone(), system };
    s.write_env(&snapshot).and_then(|()| s.append(start)).map_err(|e| format!("cannot write the session: {e}"))?;
    Ok(s)
}

/// Opens a session to continue it: a cut-off last line is dropped, unanswered tool calls get a result, and it
/// reads as running again. Also returns the index of the first event the repair appended.
fn open_session(dir: &Path) -> Result<(Session, usize), String> {
    let mut s = Session::open(dir).map_err(|e| format!("cannot open {}: {e}", dir.display()))?;
    let before = s.events.len();
    s.repair_dangling().and_then(|_| s.reopen()).map_err(|e| format!("cannot repair {}: {e}", dir.display()))?;
    Ok((s, before))
}

fn by_id(sessions: Vec<PathBuf>, id: &str) -> Result<PathBuf, String> {
    sessions.into_iter().find(|d| d.file_name().is_some_and(|n| n == id)).ok_or_else(|| format!("no session {id} in this project"))
}

fn tool_ctx(ctx: &Ctx, s: &Session) -> ToolCtx {
    ToolCtx {
        root: ctx.cwd.clone(),
        shell: ctx.shell.clone(),
        backups_dir: s.artifacts_dir(),
        reads: HashMap::new(),
        secret_env: vec![ctx.api_key_env.clone()],
    }
}

/// `-p`: one request, every event as a JSON line on stdout (from `session.events[startup..]`, appended before the
/// request). Exit 0 when the agent finished, 2 at the step limit, 1 otherwise.
fn headless(ctx: &Ctx, session: Session, startup: usize, model: &mut ZaiClient, mode: Mode, task: &str) -> ExitCode {
    eprintln!("session: {}", session.dir.display());
    let mut ui = headless_ui(io::stdout(), &session.events[startup..]);
    let perms = Perms::new(mode, true, session.dir.clone(), ctx.allow.clone());
    let tools = tool_ctx(ctx, &session);
    let mut agent = Agent { session, model, ui: &mut ui, perms, tools, budget: ctx.budget, cancel: &CANCEL, api_key_env: &ctx.api_key_env };
    let (status, code) = match agent.run_request(task) {
        Outcome::Done => ("done", 0),
        Outcome::StepLimit => ("step_limit", 2),
        Outcome::Interrupted => ("interrupted", 1),
        Outcome::Error(_) => ("error", 1),
    };
    if let Err(e) = agent.session.finish(status) {
        eprintln!("cannot write the session: {e}");
        return ExitCode::FAILURE;
    }
    if let Some(end) = agent.session.events.last() {
        agent.ui.event(end); // `session_end` is appended outside the agent
    }
    ExitCode::from(code)
}

/// The headless Ui, with `startup` (the events appended before the agent existed, such as `session_start`) already
/// written, so the stream holds every event this run appended.
fn headless_ui<W: Write>(out: W, startup: &[Event]) -> Headless<W> {
    let mut ui = Headless { out };
    for e in startup {
        ui.event(e);
    }
    ui
}

/// Tab completion over the slash commands. The completer's trie stops at any char that isn't alphanumeric or an
/// "inclusion", so `/` must be one, or nothing gets in.
fn slash_completer() -> DefaultCompleter {
    let mut c = DefaultCompleter::with_inclusions(&['/']).set_min_word_len(2);
    c.insert(SLASH_COMMANDS.map(String::from).to_vec());
    c
}

/// Windows: crossterm has no paste event, so a multi-line paste reaches reedline as typed keys and is submitted at
/// its first newline, with the rest still queued as input. Drain that rest and append it, so the paste stays one
/// request instead of becoming one request per line.
// ponytail: a 30 ms lull ends the paste, so a key typed within 30 ms of Enter joins the request; upgrade path: a
// custom reedline `EditMode` that sees the whole input burst.
fn with_pasted_rest(line: String) -> String {
    if !cfg!(windows) {
        return line;
    }
    let mut keys = Vec::new();
    while event::poll(Duration::from_millis(30)).unwrap_or(false) {
        match event::read() {
            Ok(event::Event::Key(k)) => keys.push(k),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let rest = keys_to_text(&keys);
    let rest = rest.trim_end_matches('\n');
    if rest.is_empty() {
        return line;
    }
    println!("{}", format!("  (+{} pasted lines)", rest.lines().count()).dim());
    format!("{line}\n{rest}")
}

/// The text that console keys type: characters, Tab, and Enter as a newline. Key releases (Windows reports them)
/// and keys that type nothing are dropped.
fn keys_to_text(keys: &[KeyEvent]) -> String {
    keys.iter()
        .filter(|k| k.kind != KeyEventKind::Release)
        .filter_map(|k| match k.code {
            KeyCode::Enter => Some('\n'),
            KeyCode::Tab => Some('\t'),
            KeyCode::Char(c) if !c.is_control() => Some(c),
            _ => None,
        })
        .collect()
}

fn line_editor(root: &Path) -> Reedline {
    let mut keys = default_emacs_keybindings();
    keys.add_binding(KeyModifiers::ALT, KeyCode::Enter, ReedlineEvent::Edit(vec![EditCommand::InsertNewline]));
    let tab = ReedlineEvent::UntilFound(vec![ReedlineEvent::Menu("completion_menu".into()), ReedlineEvent::MenuNext]);
    keys.add_binding(KeyModifiers::NONE, KeyCode::Tab, tab);
    let editor = Reedline::create()
        .with_edit_mode(Box::new(Emacs::new(keys)))
        .with_completer(Box::new(slash_completer()))
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(ColumnarMenu::default().with_name("completion_menu"))))
        // crossterm has no paste event on Windows (crossterm#737; reedline recommends exactly this switch);
        // `with_pasted_rest` keeps a multi-line paste together there
        .use_bracketed_paste(cfg!(not(windows)));
    match FileBackedHistory::with_file(HISTORY_SIZE, root.join("history.txt")) {
        Ok(history) => editor.with_history(Box::new(history)),
        Err(e) => {
            eprintln!("input history is off: {e}");
            editor
        }
    }
}

fn repl(ctx: &Ctx, mut session: Session, model: &mut ZaiClient, mode: Mode) -> ExitCode {
    let mut editor = line_editor(&ctx.root);
    let prompt = DefaultPrompt::new(DefaultPromptSegment::Empty, DefaultPromptSegment::Empty);
    let mut term = Terminal::new(false, &CANCEL);
    let mut perms = Perms::new(mode, false, session.dir.clone(), ctx.allow.clone());
    let mut tools = tool_ctx(ctx, &session);
    // a write to `session` failed, so its file may end in a partial line: no more requests until /clear or /resume
    let mut unwritable = false;
    let code = loop {
        let line = match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => with_pasted_rest(line),
            Ok(Signal::CtrlD) => break ExitCode::SUCCESS,
            Ok(_) => continue, // Ctrl+C: the line is dropped
            Err(e) => {
                eprintln!("cannot read input: {e}");
                break ExitCode::FAILURE;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        // The agent owns the session, permissions and tool state while it runs, and hands them back after.
        let Some(cmd) = parse_slash(&line) else {
            if unwritable {
                term.notice(UNWRITABLE);
                continue;
            }
            let mut agent = Agent { session, model: &mut *model, ui: &mut term, perms, tools, budget: ctx.budget, cancel: &CANCEL, api_key_env: &ctx.api_key_env };
            let outcome = agent.run_request(&line);
            Agent { session, perms, tools, .. } = agent;
            println!();
            if let Outcome::Error(e) = outcome
                && session_unwritable(&e)
            {
                unwritable = true;
                term.notice(UNWRITABLE);
            }
            continue;
        };
        let next = match cmd {
            Slash::Exit => break ExitCode::SUCCESS,
            Slash::Help => {
                println!("{HELP}");
                None
            }
            Slash::Think => {
                term.show_thinking = !term.show_thinking;
                println!("reasoning: {}", if term.show_thinking { "shown in full" } else { "one line" });
                None
            }
            Slash::Mode(None) => {
                println!("mode: {}", perms.mode.as_str());
                None
            }
            Slash::Mode(Some(m)) => {
                match m.parse::<Mode>() {
                    Ok(m) => {
                        perms.mode = m;
                        println!("mode: {}", m.as_str());
                    }
                    Err(e) => println!("{e}"),
                }
                None
            }
            Slash::Status => {
                println!("{}", status_line(&session_status(&session, ctx.budget)));
                None
            }
            Slash::Compact(_) if unwritable => {
                term.notice(UNWRITABLE); // a compaction appends to the session too
                None
            }
            Slash::Compact(focus) => {
                CANCEL.store(false, Ordering::SeqCst); // a Ctrl+C from an earlier request must not cut the summary call
                let mut agent = Agent { session, model: &mut *model, ui: &mut term, perms, tools, budget: ctx.budget, cancel: &CANCEL, api_key_env: &ctx.api_key_env };
                let result = agent.compact(focus.as_deref());
                Agent { session, perms, tools, .. } = agent;
                if let Err(e) = result {
                    term.notice(&e);
                    if session_unwritable(&e) {
                        unwritable = true;
                        term.notice(UNWRITABLE);
                    }
                }
                None
            }
            Slash::Clear => Some(new_session(ctx, perms.mode)),
            Slash::Resume(id) => Some(choose_session(ctx, &session.dir, id.as_deref()).and_then(|dir| open_session(&dir)).map(|(s, _)| s)),
            Slash::Unknown(c) => {
                println!("unknown command /{c} (/help lists them)");
                None
            }
        };
        match next {
            Some(Ok(next)) => {
                if let Err(e) = session.finish("done") {
                    eprintln!("cannot write the session: {e}");
                }
                session = next;
                unwritable = false;
                // "always" grants and read records belong to a session
                perms = Perms::new(perms.mode, false, session.dir.clone(), ctx.allow.clone());
                tools = tool_ctx(ctx, &session);
                println!("session {}", session.meta.id);
            }
            Some(Err(e)) => println!("{e}"),
            None => {}
        }
    };
    match session.finish("done") {
        Ok(()) => code,
        Err(e) => {
            eprintln!("cannot write the session: {e}");
            ExitCode::FAILURE
        }
    }
}

/// True for the error of a failed session write (from a request or `/compact`).
fn session_unwritable(error: &str) -> bool {
    error.starts_with(agent::SESSION_WRITE_FAILED)
}

/// `/status`: the whole session so far.
fn session_status(s: &Session, budget: usize) -> Status {
    let assistant = || s.events.iter().filter_map(|e| match &e.kind {
        Kind::Assistant { usage, tool_calls, .. } => Some((usage, tool_calls.len())),
        _ => None,
    });
    let cached_pct = assistant().next_back().filter(|(u, _)| u.prompt > 0).map_or(0, |(u, _)| (u.cached * 100 / u.prompt).min(100) as u8);
    Status {
        ctx_tokens: policy::estimate_tokens(&s.events),
        budget,
        tool_calls: assistant().map(|(_, n)| n).sum(),
        elapsed_ms: store::now_ms().saturating_sub(s.meta.started_ms),
        cached_pct,
    }
}

/// `/resume`: the session `id` of this project, or one picked by number from the newest; never the current one.
fn choose_session(ctx: &Ctx, current: &Path, id: Option<&str>) -> Result<PathBuf, String> {
    let sessions: Vec<PathBuf> = store::list_sessions(&ctx.root, &ctx.cwd).into_iter().filter(|d| d != current).collect();
    if let Some(id) = id {
        return by_id(sessions, id);
    }
    let shown = &sessions[..sessions.len().min(RESUME_LIST)];
    if shown.is_empty() {
        return Err("no earlier session in this project".into());
    }
    for (i, dir) in shown.iter().enumerate() {
        println!("  {:>2}  {}", i + 1, describe(dir));
    }
    print!("resume which? (number, Enter to cancel) › ");
    let _ = io::stdout().flush();
    let mut answer = String::new();
    io::stdin().read_line(&mut answer).map_err(|e| format!("cannot read the answer: {e}"))?;
    answer.trim().parse::<usize>().ok().and_then(|n| n.checked_sub(1)).and_then(|i| shown.get(i)).cloned().ok_or_else(|| "nothing resumed".into())
}

/// `<id>  <status>  <first request>`, read without `Session::open` (which would repair the files).
fn describe(dir: &Path) -> String {
    let id = dir.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let meta = fs::read(dir.join("meta.json")).ok().and_then(|b| serde_json::from_slice::<Meta>(&b).ok());
    let first = fs::read_to_string(dir.join("events.jsonl")).ok().and_then(|raw| {
        raw.lines().find_map(|l| match serde_json::from_str::<Event>(l).ok()?.kind {
            Kind::User { text } => Some(text),
            _ => None,
        })
    });
    // the user's own text, but it may hold pasted escape codes
    let first: String = first.as_deref().and_then(|t| t.lines().next()).unwrap_or("").chars().filter(|c| !c.is_control()).collect();
    format!("{id}  {:<11}  {}", meta.map_or_else(|| "?".into(), |m| m.status), tools::clip(&first, 60))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perms::Mode;
    use crate::store::test_dir;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn args_full() {
        let a = parse_args(&strs(&[
            "-p", "fix it", "--mode", "yolo", "--allow", "cargo test", "--allow", "ls", "--budget", "50000", "--cwd", "D:/x", "--model", "m",
            "--resume", "abc",
        ]))
        .unwrap();
        assert_eq!(a.prompt.as_deref(), Some("fix it"));
        assert_eq!(a.mode, Some(Mode::Yolo));
        assert_eq!(a.allow, ["cargo test", "ls"]);
        assert_eq!(a.budget, Some(50_000));
        assert_eq!(a.cwd, Some(PathBuf::from("D:/x")));
        assert_eq!(a.model.as_deref(), Some("m"));
        assert_eq!(a.resume, Some(Some("abc".to_string())));
        assert_eq!(parse_args(&[]).unwrap(), Args::default());
    }

    #[test]
    fn args_resume_without_id_and_errors() {
        assert_eq!(parse_args(&strs(&["--resume"])).unwrap().resume, Some(None));
        let a = parse_args(&strs(&["--resume", "-p", "x"])).unwrap();
        assert_eq!((a.resume, a.prompt.as_deref()), (Some(None), Some("x")));
        assert_eq!(parse_args(&strs(&["--x"])), Err("unknown flag --x".to_string()));
        assert!(parse_args(&strs(&["--mode", "bad"])).is_err());
        assert!(parse_args(&strs(&["--budget", "abc"])).is_err());
        assert!(parse_args(&strs(&["-p"])).is_err());
        assert!(parse_args(&strs(&["--allow"])).is_err());
    }

    #[test]
    fn args_version_flag() {
        assert!(!Args::default().version);
        assert!(parse_args(&strs(&["--version"])).unwrap().version);
        // it can sit anywhere and needs no value
        assert!(parse_args(&strs(&["--mode", "ask", "--version", "-p", "x"])).unwrap().version);
        assert_eq!(parse_args(&strs(&["--version"])).unwrap(), Args { version: true, ..Args::default() });
    }

    #[test]
    fn config_defaults_partial_and_invalid() {
        let dir = test_dir("config");
        let d = load_config(&dir.join("missing.json")).unwrap();
        assert_eq!(d.base_url, "https://api.z.ai/api/coding/paas/v4");
        assert_eq!(d.model, "glm-5.3-flash");
        assert_eq!(d.api_key_env, "ZAI_API_KEY");
        assert_eq!(d.context_budget, 200_000);
        assert_eq!(d.mode, "writes");
        assert_eq!((d.temperature, d.top_p), (1.0, 0.95));
        assert_eq!(d.reasoning_effort.as_deref(), Some("max"));

        let partial = dir.join("partial.json");
        fs::write(&partial, r#"{"context_budget":50000}"#).unwrap();
        let p = load_config(&partial).unwrap();
        assert_eq!(p.context_budget, 50_000);
        assert_eq!((p.model.as_str(), p.mode.as_str()), ("glm-5.3-flash", "writes"));

        // Windows editors and PowerShell's `Set-Content -Encoding utf8` write a BOM
        let bom = dir.join("bom.json");
        fs::write(&bom, "\u{feff}{\"model\":\"m\"}").unwrap();
        assert_eq!(load_config(&bom).unwrap().model, "m");

        let bad = dir.join("bad.json");
        fs::write(&bad, "{bad").unwrap();
        let err = load_config(&bad).unwrap_err();
        assert!(err.contains(&bad.display().to_string()), "{err}");

        // a misspelled key is an error, not a silently ignored setting
        let typo = dir.join("typo.json");
        fs::write(&typo, r#"{"reasoning_efort":"low"}"#).unwrap();
        let err = load_config(&typo).unwrap_err();
        assert!(err.contains("reasoning_efort") && err.contains(&typo.display().to_string()), "{err}");
    }

    #[test]
    fn flags_over_config_and_validation() {
        let args = parse_args(&strs(&["--model", "m2", "--budget", "1000", "--mode", "ask"])).unwrap();
        let (cfg, mode) = apply(Config::default(), &args).unwrap();
        assert_eq!((cfg.model.as_str(), cfg.context_budget, mode), ("m2", 1000, Mode::Ask));
        assert_eq!(apply(Config::default(), &Args::default()).unwrap().1, Mode::Writes);

        let bad_effort = Config { reasoning_effort: Some("MAX".into()), ..Config::default() };
        let err = apply(bad_effort, &Args::default()).unwrap_err();
        assert!(err.contains("MAX") && err.contains("max, xhigh, high, medium, low, minimal, none"), "{err}");
        for e in ["max", "xhigh", "high", "medium", "low", "minimal", "none"] {
            assert!(apply(Config { reasoning_effort: Some(e.into()), ..Config::default() }, &Args::default()).is_ok(), "{e}");
        }
        assert!(apply(Config { reasoning_effort: None, ..Config::default() }, &Args::default()).is_ok());
        assert!(apply(Config { mode: "careful".into(), ..Config::default() }, &Args::default()).is_err());
        // a flag wins over a bad config mode
        assert!(apply(Config { mode: "careful".into(), ..Config::default() }, &args).is_ok());
        assert!(apply(Config { context_budget: 0, ..Config::default() }, &Args::default()).is_err());
    }

    #[test]
    fn api_key_is_trimmed_and_falls_back_to_the_user_scope() {
        let none = || None;
        assert_eq!(pick_key(Some(" sk-1\r\n".into()), none).as_deref(), Some("sk-1"));
        assert_eq!(pick_key(Some("  ".into()), || Some("\tsk-2 ".into())).as_deref(), Some("sk-2"));
        assert_eq!(pick_key(None, || Some("sk-3".into())).as_deref(), Some("sk-3"));
        assert_eq!(pick_key(Some("sk-4".into()), || panic!("the process value wins")).as_deref(), Some("sk-4"));
        assert_eq!(pick_key(None, || Some(" ".into())), None);
        assert_eq!(pick_key(None, none), None);

        let out = "\r\nHKEY_CURRENT_USER\\Environment\r\n    ZAI_API_KEY    REG_SZ    abc.def 123\r\n\r\n";
        assert_eq!(reg_value(out, "ZAI_API_KEY").as_deref(), Some("abc.def 123"));
        assert_eq!(reg_value(out, "zai_api_key").as_deref(), Some("abc.def 123")); // Windows names are case-insensitive
        assert_eq!(reg_value(out, "OTHER"), None);
        assert_eq!(reg_value("    K    REG_EXPAND_SZ    v\r\n", "K").as_deref(), Some("v"));
        assert_eq!(reg_value("", "K"), None);
    }

    #[test]
    fn project_root_is_absolute_and_plain() {
        let dir = test_dir("root");
        fs::create_dir_all(dir.join("sub")).unwrap();
        let root = project_root(Some(&dir.join("sub").join(""))).unwrap(); // trailing separator
        assert_eq!(root, dir.join("sub"));
        assert!(root.is_absolute());
        assert!(!root.to_string_lossy().starts_with(r"\\?\"), "{}", root.display());
        assert!(project_root(Some(&dir.join("nope"))).is_err());
        fs::write(dir.join("f.txt"), "x").unwrap();
        assert!(project_root(Some(&dir.join("f.txt"))).is_err());
        assert!(project_root(None).unwrap().is_absolute());
        assert!(project_root(Some(Path::new("."))).unwrap().is_absolute());
    }

    #[test]
    fn slash_parsing() {
        assert_eq!(parse_slash("/mode yolo"), Some(Slash::Mode(Some("yolo".into()))));
        assert_eq!(parse_slash("/mode"), Some(Slash::Mode(None)));
        assert_eq!(parse_slash("/mode   "), Some(Slash::Mode(None)));
        assert_eq!(parse_slash("/compact focus on parser"), Some(Slash::Compact(Some("focus on parser".into()))));
        assert_eq!(parse_slash("/exit"), Some(Slash::Exit));
        assert_eq!(parse_slash("/nope"), Some(Slash::Unknown("nope".into())));
        assert_eq!(parse_slash("hello"), None);
        for (line, want) in [
            ("/help", Slash::Help),
            ("/clear", Slash::Clear),
            ("/resume", Slash::Resume(None)),
            ("/resume 20261004-091500-a3f2", Slash::Resume(Some("20261004-091500-a3f2".into()))),
            ("/think", Slash::Think),
            ("/status", Slash::Status),
        ] {
            assert_eq!(parse_slash(line), Some(want), "{line}");
        }
        // every completer entry is a command parse_slash knows
        for name in SLASH_COMMANDS {
            assert!(!matches!(parse_slash(name), Some(Slash::Unknown(_)) | None), "{name}");
        }
    }

    #[test]
    fn slash_commands_complete() {
        use reedline::Completer;
        let mut c = slash_completer();
        let mut values = |line: &str| c.complete(line, line.len()).suggestions().iter().map(|s| s.value.clone()).collect::<Vec<_>>();
        assert_eq!(values("/he"), ["/help"]);
        assert_eq!(values("/c"), ["/clear", "/compact"]);
        assert_eq!(values("/re"), ["/resume"]);
        assert!(values("hel").is_empty());
    }

    fn test_session(tag: &str) -> Session {
        let dir = test_dir(tag);
        let meta = Meta {
            id: "20260101-000000-0001".into(),
            cwd: dir.join("proj").display().to_string(),
            git_head: None,
            model: "m".into(),
            mode: "writes".into(),
            policy_hash: "h".into(),
            started_ms: 0,
            ended_ms: None,
            status: "running".into(),
        };
        let mut s = Session::create(&dir.join("home"), meta).unwrap();
        s.append(Kind::SessionStart { cwd: "/w".into(), model: "m".into(), mode: "writes".into(), policy_hash: "h".into(), system: "S".into() })
            .unwrap();
        s
    }
    fn streamed(out: &[u8]) -> Vec<Event> {
        std::str::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    #[test]
    fn headless_stream_starts_with_session_start() {
        // a new session: `session_start` was appended before the Ui existed, and still opens the stream
        let s = test_session("headless-new");
        let ui = headless_ui(Vec::new(), &s.events[0..]);
        let lines = streamed(&ui.out);
        assert_eq!(lines, s.events);
        assert!(matches!(lines[0].kind, Kind::SessionStart { .. }), "{lines:?}");

        // a resumed session: only what the repair appended at startup (a result for the dangling call), not the old history
        let mut s = test_session("headless-resume");
        s.append(Kind::User { text: "go".into() }).unwrap();
        let call = llm::ToolCall { id: "c1".into(), name: "read".into(), args: "{}".into() };
        let reply = Kind::Assistant { text: String::new(), reasoning: String::new(), tool_calls: vec![call], usage: llm::Usage::default(), finish_reason: "tool_calls".into(), latency_ms: 0 };
        s.append(reply).unwrap();
        let (s, startup) = open_session(&s.dir).unwrap();
        assert_eq!(startup, 3);
        let lines = streamed(&headless_ui(Vec::new(), &s.events[startup..]).out);
        assert_eq!(lines, s.events[3..]);
        assert!(matches!(&lines[..], [Event { kind: Kind::ToolResult { call_id, .. }, .. }] if call_id == "c1"), "{lines:?}");
    }

    #[test]
    fn a_failed_session_write_is_recognized() {
        assert!(session_unwritable(&format!("{}: disk full", agent::SESSION_WRITE_FAILED)));
        assert!(!session_unwritable("network error: reset"));
        assert!(!session_unwritable("compaction failed: cannot write summary: x"));
    }

    #[test]
    fn pasted_keys_become_text() {
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
        let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
        let release = |code| KeyEvent::new_with_kind(code, KeyModifiers::NONE, KeyEventKind::Release);
        let shifted = KeyEvent::new(KeyCode::Char('B'), KeyModifiers::SHIFT);
        let keys = [
            release(KeyCode::Enter), // the submitting Enter's release comes first
            press(KeyCode::Char('a')),
            release(KeyCode::Char('a')),
            press(KeyCode::Enter),
            press(KeyCode::Tab),
            shifted,
            press(KeyCode::Char(' ')),
            press(KeyCode::Char('é')),
            press(KeyCode::Left), // not text
            press(KeyCode::Enter),
        ];
        assert_eq!(keys_to_text(&keys), "a\n\tB é\n");
        assert_eq!(keys_to_text(&[release(KeyCode::Enter)]), "");
        assert_eq!(keys_to_text(&[]), "");
    }
}

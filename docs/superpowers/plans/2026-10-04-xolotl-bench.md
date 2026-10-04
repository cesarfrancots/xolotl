# Xolotl Bench (Local Suite) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 12-task local benchmark for Xolotl, with a dependency-free runner, hidden checks and committed results history. Then record the v1 baseline with 3 repeats.

**Architecture:**
- `bench/run.py` is a Python standard-library script. Pure functions (slug, task loading, metrics, history rows) are kept separate from orchestration (build, temp dirs, subprocess with tree-kill, verify).
- The orchestration takes the Xolotl command as a list, so tests drive it with a fake Xolotl. No model calls.
- Each task is a folder holding `task.json`, `repo/`, `verify/` (hidden) and `solution/` (reference). `--selfcheck` proves every task is solvable and not already solved.

**Tech Stack:** Python 3.10+ stdlib only (3.14 installed), `unittest`. Tasks use Python, Rust/cargo, Node and Git Bash.

**Spec:** `docs/superpowers/specs/2026-10-04-xolotl-bench-design.md`

## Global Constraints

- **Dependencies:** Python standard library only. No pip packages, no Docker.
- **Platform:** Windows 11 primary, with Git Bash present. On Windows, Bash is resolved from `git`'s location (`<git>\..\..\bin\bash.exe`), never `System32\bash.exe`. Other OSes use `bash`.
- **Secrets:** the runner never reads, prints or stores the API key or any environment variable. Xolotl reads the key itself.
- **Fairness:**
  - `verify/` and `solution/` are never visible to the agent.
  - Before verifying, the runner deletes any `<tmp>/verify` the agent created, then copies the real one in.
- **Determinism:** hidden checks are deterministic. Expected outputs are computed from the inputs inside `verify/`, not hard-coded, wherever possible.
- **Toolchains:** a task lists them in `requires`; if one is missing the task is `skipped`, which doesn't count as a failure.
- **Git:**
  - Branch `feat/bench`, stacked on `feat/xolotl-v1`.
  - Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Commit steps below show only the subject line.
- **Tests:** `bench/test_run.py` (`unittest`), run with `py -m unittest bench.test_run -v` from the repo root. Add `bench/__init__.py` so the module path resolves.

## Review Focus

1. **A timed-out Xolotl leaves `bash`/`cargo` children running,** which then poison later tasks. Pinned by `timeout_kills_tree_and_returns_promptly` (Task 2).
2. **The agent pre-plants a passing file at `verify/`.** It must be wiped before verification. Pinned by `planted_verify_is_replaced` (Task 2).
3. **Xolotl stdout that is huge, non-UTF-8 or truncated** (killed mid-line) must not crash metrics extraction. Pinned by `metrics_tolerate_garbage_lines` (Task 1).
4. **A task that is accidentally already solved, or unsolvable,** would make the baseline meaningless. Pinned by `--selfcheck` over every task (Tasks 3 and 4).
5. **The session-folder lookup must pick this attempt's session,** not an older one from the same slug. Pinned by `session_dir_picks_newest` (Task 1).

---

### Task 1: Runner pure functions

**Files:**
- Create: `bench/__init__.py` (empty), `bench/run.py`, `bench/test_run.py`

**Interfaces:**
- Produces, in `bench/run.py`:
```python
def slug(path: str) -> str            # same rule as Rust store::project_slug
def load_task(task_dir: Path) -> dict  # validated task.json; raises ValueError naming the missing/bad field
def metrics_from_events(lines: Iterable[str]) -> dict
def session_dir(home: Path, cwd: str) -> Path | None   # newest ~/.xolotl/projects/<slug>/sessions/*
def summarize(attempts: list[dict]) -> dict
def history_row(result: dict) -> str   # one Markdown table row
HISTORY_HEADER: str                    # header + separator lines of results/HISTORY.md
```
- `task.json` fields:

  | Field | Kind | Default |
  |---|---|---|
  | `id` | required str | — |
  | `title` | required str | — |
  | `tags` | list[str] | `[]` |
  | `prompt` | required str | — |
  | `verify` | required non-empty list[str] | — |
  | `timeout_s` | int | 600 |
  | `verify_timeout_s` | int | 120 |
  | `budget` | int or null | null |
  | `requires` | list[str] of executables | `[]` |

  `id` must equal the folder name.
- `metrics_from_events` returns:
  - `model_calls`, `tool_calls`, `tokens{prompt, completion, cached}`
  - `compactions{stage1, stage2}`, `completion_check: bool`, `errors`, `policy_hash`, `interrupted: bool`
  - It ignores lines that aren't JSON objects.
- `summarize` returns:
  - `pass`, `total`, `skipped`, `pass_rate` (pass / (total − skipped), rounded to 3 places)
  - `per_task{id: "k/n"}`, `tokens{…sums}`, `median_wall_s`, `total_wall_s`

- [ ] **Step 1: Write the failing tests in `bench/test_run.py`**

```python
class Pure(unittest.TestCase):
    def test_slug_matches_rust(self):
        self.assertEqual(run.slug(r"D:\Xolotl Harness"), "D--Xolotl-Harness")
        self.assertEqual(run.slug("/home/me/my proj"), "-home-me-my-proj")
    def test_load_task_validates(self):
        # a good task.json loads with defaults filled (timeout_s 600, verify_timeout_s 120, budget None, requires [])
        # missing "prompt" → ValueError mentioning "prompt"; empty verify → ValueError; id != folder name → ValueError
    def test_metrics_from_events(self):
        # canned lines:
        #   session_start(policy_hash "abc"),
        #   assistant(usage 100/20/50), tool_result,
        #   assistant(usage 200/30/0), tool_result,
        #   compaction stage 1, compaction stage 2,
        #   nudge completion_check, error, interrupt
        # → model_calls 2, tool_calls 2, tokens {300, 50, 50}, stage1 1, stage2 1,
        #   completion_check True, errors 1, policy_hash "abc", interrupted True
    def test_metrics_tolerate_garbage_lines(self):
        # "", "not json", '{"type":"assistant"' (truncated), b"\xff".decode("latin-1"), "[1,2]"
        # mixed with one valid assistant line → model_calls 1, no exception
    def test_session_dir_picks_newest(self):
        # home/.xolotl/projects/<slug>/sessions/{20261004-100000-aaaa, 20261004-110000-bbbb} → the 1100 one;
        # missing project dir → None
    def test_summarize_and_history_row(self):
        # 4 attempts: t1 pass, t1 fail, t2 pass, t3 skipped → pass 2, total 4, skipped 1, pass_rate 0.667,
        # per_task {"t1": "1/2", "t2": "1/1", "t3": "skipped"}
        # history_row(result) == "| 2026-10-04 15:30 | 20261004-1530-82ad7b2 | abcdef12 | 2/3 (66.7%) | 300 / 50 | 12.0s | v1 baseline |"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m unittest bench.test_run -v`
Expected: errors, because `run` has no such attributes.

- [ ] **Step 3: Implement the pure functions in `bench/run.py`**

- **`slug`:** any char that isn't ASCII alphanumeric, `-`, `_` or `.` becomes `-`.
- **`metrics_from_events`:** `json.loads` each line inside try/except, keeping only `dict`s.
- **`history_row`:**
  - Date comes from `result["started"]` (ISO), formatted `YYYY-MM-DD HH:MM`.
  - Policy hash: the first 8 characters.
  - Pass shown as `k/N (x.x%)`, where N excludes skipped tasks.
  - Tokens shown as `prompt / completion`.
  - Median wall time with one decimal.
  - Then the note.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m unittest bench.test_run -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add bench/
git commit -m "feat(bench): add runner pure functions with tests"
```

---

### Task 2: Runner orchestration, selfcheck and README

**Files:**
- Modify: `bench/run.py`, `bench/test_run.py`
- Create: `bench/README.md`, `bench/results/HISTORY.md` (header only), `bench/tests_fixtures/fake_xolotl.py`, `bench/tests_fixtures/tasks/fx-ok/…`

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
```python
def find_bash() -> list[str]                       # argv prefix for running a verify command
def run_attempt(task: dict, task_dir: Path, xolotl_cmd: list[str], home: Path,
                budget: int | None, keep: bool) -> dict
def selfcheck(task: dict, task_dir: Path) -> tuple[bool, str]   # (ok, reason)
def main(argv: list[str]) -> int
```
- **CLI:** `py bench/run.py [--tasks a,b] [--repeat N] [--budget N] [--xolotl PATH] [--note TEXT] [--keep] [--selfcheck] [--tasks-dir DIR]`
  - `--tasks-dir` defaults to `bench/tasks`.
  - `--xolotl PATH` may be an executable, or a `.py` file, which runs with `sys.executable`.
- **One attempt's dict:**
  - `task`, `attempt`, `status` (`pass` | `fail` | `skipped`), `pass`
  - `exit_code`, `timed_out`, `wall_s`
  - `verify_output` (last 2 KB), `kept_dir`, `session_dir`
  - every key from `metrics_from_events`

- [ ] **Step 1: Write the failing tests**

`fake_xolotl.py` reads `--cwd` and `-p` from argv. Its behavior is driven by the prompt text:
- `"ok"` → writes `answer.txt` containing `42` and prints 3 JSON event lines (`session_start`, `assistant` with usage, `session_end`), exit 0.
- `"plant"` → writes `verify/test_planted.py` that always passes, plus nothing else, exit 0.
- `"hang"` → spawns a child `python -c "import time; time.sleep(60)"` and sleeps 60 s itself.
- `"garbage"` → prints 1 MB of non-JSON bytes, then one assistant line.

Fixture task `fx-ok`:
- `repo/` holds `README.txt`.
- `verify/check.py` asserts that `answer.txt` == `42`.
- `solution/answer.txt` contains `42`.
- `task.json` has `verify: ["{python} verify/check.py"]`.

```python
class Orchestration(unittest.TestCase):
    def test_pass_path(self):            # prompt ok → status pass, model_calls 1, temp dir deleted (kept_dir None)
    def test_planted_verify_is_replaced(self):   # prompt plant → status fail (planted file gone, real check fails)
    def test_timeout_kills_tree_and_returns_promptly(self):
        # prompt hang, timeout_s 3 → timed_out True, status fail, run_attempt returns in < 15 s,
        # and the spawned child PID is no longer alive (record it via a pid file the fake writes)
    def test_garbage_stdout(self):       # prompt garbage → no exception, model_calls 1
    def test_missing_toolchain_skips(self):   # requires ["definitely-not-a-real-exe"] → status skipped, no subprocess run
    def test_selfcheck(self):            # fx-ok → (True, ...); a variant whose solution doesn't fix it → (False, "solution fails …");
                                         # a variant already passing without the solution → (False, "unsolved repo already passes")
    def test_main_writes_results_and_history(self):
        # main([... "--tasks-dir", fixtures, "--xolotl", fake, "--note", "t"]) in a temp results dir
        # → results/<run_id>.json has summary.pass == 1, and HISTORY.md gained exactly one row ending with "| t |"
```

The tests pass a temp `home` and a temp results dir. Add a `--results-dir` hidden option, default `bench/results`, so tests never touch the real files.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m unittest bench.test_run -v`
Expected: the orchestration tests error.

- [ ] **Step 3: Implement**

**`run_attempt`**:
1. **Toolchains:** if any `shutil.which(r)` in `requires` is missing, return `skipped`.
2. **Temp dir:** `tempfile.mkdtemp(prefix="xolotl-bench-")`. Copy `repo/` in, then run `git init`, `git add -A` and `git -c user.name=bench -c user.email=bench@local commit -qm init`.
3. **Run Xolotl:** `xolotl_cmd + ["-p", prompt, "--mode", "yolo", "--cwd", tmp] (+ ["--budget", N])` via `subprocess.Popen`.
   - Pass `stdout=PIPE, stderr=PIPE` and read both through threads, so a full pipe can't deadlock the child.
   - On Windows use `creationflags=CREATE_NEW_PROCESS_GROUP`; on POSIX use `start_new_session=True`.
4. **Timeout:** kill the tree with `taskkill /T /F /PID <pid>` on Windows or `os.killpg(pgid, SIGKILL)` elsewhere, then wait. Decode output with `errors="replace"`.
5. **Verify:**
   - `shutil.rmtree(tmp/verify, ignore_errors=True)`, then `copytree(task_dir/verify, tmp/verify)`.
   - Run each `verify` command, with `{python}` replaced by `sys.executable`, through `find_bash() + ["-c", cmd]` with `cwd=tmp` and `verify_timeout_s`.
   - Pass means every command exits 0.
6. **Session:** `session_dir(home, tmp)`.
7. **Cleanup:** delete `tmp` unless `keep` is set or the status is not `pass`.

**`selfcheck`:**
- Copy `repo/`, overlay `solution/`, copy `verify/`, run verify. It must pass.
- Then copy `repo/` alone plus `verify/`. That must fail.
- Model calls: none.

**`main`:**
1. Parse arguments and pick tasks (sorted folder names, or `--tasks`).
2. If `--selfcheck`: print one line per task and exit 1 if any fails.
3. Otherwise:
   1. Build with `cargo build --release` unless `--xolotl` was given, and copy the exe into a temp dir.
   2. Get the git SHA (`git rev-parse HEAD`), the dirty flag (`git status --porcelain` is non-empty), `xolotl --version` and `started` (now, ISO).
   3. Run `repeat × tasks` attempts, printing one progress line each.
   4. Write `results/<run_id>.json` (the spec §6 schema plus `started`), and append `history_row`, writing `HISTORY_HEADER` first if the file is empty.
4. **Ctrl+C** (`KeyboardInterrupt`): kill the current child tree, write the results with `"partial": true`, don't append to history, and return 130.

**`find_bash`:**
- On Windows: `git` from `shutil.which("git")` → `<git>/../../bin/bash.exe` if it exists, else `["cmd", "/c"]`.
- Elsewhere: `["bash"]`.

`bench/README.md`: how to run (`py bench/run.py --selfcheck`, a baseline run, a single task), how to add a task (the folder layout and `task.json` fields), and what the metrics mean. Keep it short.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m unittest bench.test_run -v`
Expected: all pass. The timeout test finishes in about 3–5 s.

- [ ] **Step 5: Commit**

```bash
git add bench/
git commit -m "feat(bench): add runner orchestration, selfcheck and README"
```

---

### Task 3: Python tasks (7)

**Files:**
- Create: `bench/tasks/{py-offbyone,py-docstring,py-rename,py-argparse,py-logparse,py-crossmodule,py-perf}/{task.json,repo/…,verify/…,solution/…}`

**Interfaces:**
- Consumes: Task 2's `--selfcheck`.
- All 7 have `requires: []`, since Python is guaranteed. Hidden checks are `unittest` files run as `{python} -m unittest discover -s verify -v`, unless stated otherwise.
- Prompts are exact (copy them into `task.json` verbatim).

| id | `repo/` contents | prompt | hidden check (`verify/`) | `solution/` |
|---|---|---|---|---|
| `py-offbyone` | `stats.py`: `moving_average(xs, k)` returns the list of means of each length-k window. Docstring: raises `ValueError` if k ≤ 0; returns `[]` if k > len(xs). Bug: `range(len(xs) - k)` instead of `+ 1`. `tests/test_stats.py` has one visible failing test. | `The test in tests/test_stats.py fails. Find and fix the bug in the code without changing the tests.` | `[1,2,3,4]`,k=2 → `[1.5,2.5,3.5]`; k=4 → `[2.5]`; k=1 → identity as floats; k=5 → `[]`; k=0 → ValueError | fixed `stats.py` |
| `py-docstring` | `text_utils.py`: `def slugify(text: str) -> str:` with this docstring and a `raise NotImplementedError` body: lowercase; strip accents (NFKD, drop combining marks); replace each run of non-alphanumeric chars with `-`; trim `-` from both ends | `Implement slugify in text_utils.py exactly as its docstring describes.` | `"Hello, World!"→"hello-world"`, `"  Crème Brûlée  "→"creme-brulee"`, `"a--b__c"→"a-b-c"`, `""→""`, `"123 Go"→"123-go"`, `"Ünïcödé"→"unicode"`, `"---"→""` | implemented `slugify` |
| `py-rename` | package `shop/` with `__init__.py`, `users.py` (defines `get_usr(uid)`), `orders.py` and `reports.py` (both call it); `tests/test_shop.py` calls it | `Rename the function get_usr to get_user everywhere in this project (definition, imports, call sites and tests). Behavior must not change.` | no `.py` file outside `verify/` contains `get_usr`; `shop.users.get_user` exists; `orders`/`reports` behavior is unchanged (hidden tests call them) | the renamed files |
| `py-argparse` | `top.py`: reads `scores.csv` (name,score) and prints `name score` lines, sorted by score descending, using argparse with a positional path | `Add a --limit N option to top.py that prints only the top N rows. N must be a positive integer; reject invalid values with a non-zero exit code.` | runs with no flag (all 8 rows), `--limit 3` (3 rows, correct order), `--limit 0` and `--limit x` (non-zero exit) | updated `top.py` |
| `py-logparse` | `access.log`: 60 Common Log Format lines (mixed statuses 200/301/404/500, about 8 paths with deliberate ties) | `Write summarize.py that reads access.log and writes summary.json containing: "total" (number of requests), "by_status" (object mapping each HTTP status code as a string to its count) and "top_paths" (list of the 3 most requested paths as [path, count] pairs, most requested first, ties broken alphabetically). Run it once.` | runs `summarize.py` fresh, then compares `summary.json` to the expected values computed by parsing `access.log` inside the check | `summarize.py` |
| `py-crossmodule` | `pricing.py`: `apply_discount(price, pct)` with the bug `price * pct` instead of `price * (1 - pct)`, pct a fraction. `orders.py`: `order_total(items, pct)` uses it. `tests/test_orders.py` fails. | `tests/test_orders.py fails. Fix the underlying bug without changing the tests.` | direct `apply_discount` cases (0, 0.25, 1.0) plus `order_total` cases | fixed `pricing.py` |
| `py-perf` | `fast.py`: `dedupe(items)` keeps the first occurrence, preserving order, implemented as `if x not in out: out.append(x)` | `dedupe() in fast.py is too slow for large inputs: it must handle 200,000 items in well under 2 seconds. Make it fast without changing its behavior.` | correctness on small lists (order kept, mixed ints and strings); 200k ints with 50% duplicates take under 2.0 s | `seen` set version |

- [ ] **Step 1: Create the 7 task folders.**

For each, write `repo/`, `verify/` and `solution/`, plus `task.json` with the exact prompt.

- [ ] **Step 2: Run the selfcheck**

Run: `py bench/run.py --selfcheck --tasks py-offbyone,py-docstring,py-rename,py-argparse,py-logparse,py-crossmodule,py-perf`
Expected: 7 lines ending in `ok`, and exit 0.

- [ ] **Step 3: Commit**

```bash
git add bench/tasks/
git commit -m "feat(bench): add 7 Python tasks"
```

---

### Task 4: Rust, Node, environment and long-context tasks (5)

**Files:**
- Create: `bench/tasks/{rs-compile,rs-method,js-fix,env-tool,long-context}/…`

| id | requires | `repo/` contents | prompt | hidden check | `solution/` |
|---|---|---|---|---|---|
| `rs-compile` | `cargo` | crate `calc` (no deps). `src/lib.rs` has a type mismatch (returns `i64` where the signature says `u32`) and a borrow error (pushing into a Vec while iterating it). `tests/calc.rs` has integration tests. | ``cargo test fails to compile in this crate. Make `cargo test` pass without modifying anything under tests/.`` | `verify/check.py`: the sha256 of `tests/calc.rs` equals the hash recorded in `verify/hashes.json`, and `cargo test --quiet` exits 0. Command: `{python} verify/check.py`. | fixed `src/lib.rs` |
| `rs-method` | `cargo` | crate `stack`: `Stack<T>` with `new`, `push`, `pop`, `len`, `is_empty`, plus unit tests in `src/lib.rs` | ``Add a peek(&self) -> Option<&T> method to Stack that returns the top element without removing it, and add a unit test for it.`` | copy `verify/peek_hidden.rs` to `tests/peek_hidden.rs` and run `cargo test --quiet`; also require a `#[test]` fn whose body mentions `peek` in `src/` | the `src/lib.rs` with `peek` and its test |
| `js-fix` | `node` | `dates.js` exporting `formatDuration(seconds)`. Doc comment: `3723 → "1h 2m 3s"`, `60 → "1m"`, `0 → "0s"`, zero units omitted. Bug: minutes computed as `Math.floor(seconds / 60)` without `% 60`. | `formatDuration in dates.js returns wrong results for some inputs. Fix it; the expected format is described in its doc comment.` | `node verify/check.js` with `assert.strictEqual` on 0, 59, 60, 61, 3600, 3723, 86399 | fixed `dates.js` |
| `env-tool` | `[]` | `data/` with 5 small `.txt` files (names chosen to sort non-trivially, e.g. `b.txt`, `a10.txt`, `a2.txt`) plus one `notes.md` | `Write a script named report.py or report.sh (your choice) that prints one line per .txt file in data/, sorted by file name, formatted as "<sha256 hex>  <file name>" (two spaces). Run it once to show the output.` | `verify/check.py`: runs `report.py` with Python if present, else `report.sh` via bash; compares stdout lines to hashlib values in plain lexicographic name order | `report.py` |
| `long-context` | `[]`; `budget: 30000` | `config.py` defines `CONFIG = {}`. 25 modules `mod_01.py`…`mod_25.py`, about 40 lines each, 3–5 functions each. Exactly 9 functions write to the global `CONFIG` (key assignment, `del`, `.update`, `.pop`, `.setdefault`, `.clear`); others only read it; 3 modules have a *local* dict named `config` that they mutate (decoys). | `Across all modules in this project, find every function that WRITES to the global CONFIG dict from config.py (assigning a key, del, update, pop, setdefault, clear — reading doesn't count). List them in FINDINGS.md, one per line, as module:function (for example mod_03:load_defaults), sorted alphabetically.` | `verify/check.py`: extracts all `mod_\d\d:\w+` tokens from `FINDINGS.md` and compares the set to `verify/expected.txt` exactly | `FINDINGS.md` |

- [ ] **Step 1: Create the 5 task folders as specified.**

Generate the long-context modules with a deterministic script that you run once. Commit only the output, not the generator.

- [ ] **Step 2: Run the selfcheck over all tasks**

Run: `py bench/run.py --selfcheck`
Expected: 12 lines ending in `ok`, and exit 0.

- [ ] **Step 3: Commit**

```bash
git add bench/tasks/
git commit -m "feat(bench): add Rust, Node, environment and long-context tasks"
```

---

### Task 5: v1 baseline run

**Files:**
- Create: `bench/results/<run_id>.json`
- Modify: `bench/results/HISTORY.md`

- [ ] **Step 1: Run the baseline (live, uses quota; about 1 hour)**

Run: `py bench/run.py --repeat 3 --note "v1 baseline"`
Expected:
- the run completes (exit 0)
- a results file appears
- `HISTORY.md` gains one row
- no task errors out of the runner (any status of pass, fail or skipped is fine)

- [ ] **Step 2: Sanity-check the results**

- every attempt has a `session_dir`
- `policy_hash` is identical across attempts
- `dirty` is false (commit everything first)
- `summary.total == 36`

Note any task that fails 3/3. That is useful signal about Xolotl, and it is not a bug in the bench.

- [ ] **Step 3: Commit**

```bash
git add bench/results/
git commit -m "bench: record v1 baseline (3 repeats)"
```

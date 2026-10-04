# Xolotl Bench — Local Suite Design Spec (sub-project 1)

- **Date:** 2026-10-04
- **Status:** Draft, awaiting review
- **Depends on:** Xolotl v1 (`feat/xolotl-v1`, PR #1)
- **Next:** sub-project 2 is a Terminal-Bench 2.0 adapter via Harbor (separate spec).

## 1. Purpose

Record how Xolotl performs today, in a form later versions can be compared against. Each run produces a results file tagged with:

- the git SHA,
- `policy_hash`, the hash of `src/policy.rs` that every session already records,
- the model.

So when v2 (a Meta-Harness-style loop that rewrites `policy.rs`) or any manual change lands, re-running the suite answers one question: is it better, and at what token cost?

The suite also doubles as v2's search set: small, fast, discriminative, and its checks can't be gamed (paper App. D).

## 2. Goals and non-goals

**Goals**

1. A suite of 12 tasks that runs natively on Windows in about 15–25 minutes per repeat.
2. A runner with no new dependencies (Python standard library only; `py` is installed).
3. Per-task metrics and a run summary written to `bench/results/`, plus one appended row in `HISTORY.md`.
4. Checks that the agent cannot see or modify.
5. A committed v1 baseline, run with 3 repeats.

**Non-goals**

- Terminal-Bench 2.0, Docker and Harbor (sub-project 2).
- The v2 search loop itself.
- Committing raw traces (they stay in `~/.xolotl/`).
- Parallel task execution.
- Network access for tasks beyond the model API.

## 3. Layout

```
bench/
  run.py                        the runner
  README.md                     how to run, how to add a task
  tasks/<id>/task.json          task definition (§4)
  tasks/<id>/repo/              starting files the agent works in
  tasks/<id>/verify/            hidden checks, copied in only after the agent finishes
  tasks/<id>/solution/          reference solution, used only by --selfcheck (§9)
  results/<run_id>.json         one file per run (§6)
  results/HISTORY.md            one row per run (§6)
```

`run_id` = `YYYYMMDD-HHMM-<sha7>`, in local time.

## 4. Task format

`task.json`:

```json
{
  "id": "py-offbyone",
  "title": "Fix an off-by-one guided by a failing test",
  "tags": ["python", "bugfix"],
  "prompt": "The test in tests/test_stats.py fails. Find and fix the bug.",
  "verify": ["{python} -m unittest discover -s verify -v"],
  "timeout_s": 600,
  "verify_timeout_s": 120,
  "budget": null,
  "requires": []
}
```

- `{python}` in a verify command is replaced with the runner's own interpreter (`sys.executable`), so tasks aren't tied to `py` or `python3`.
- `verify` commands run in the task's temp directory after `verify/` has been copied in.
  - They run through `bash -c` when Git Bash is present, otherwise through `cmd /c`.
  - The task passes only if every command exits 0.
- `budget`, when set, is passed as `--budget` to Xolotl. It overrides the runner's `--budget`.
- `requires` lists executables the task needs (e.g. `["cargo"]`). If any is missing from PATH, the task is `skipped`.
- `repo/` must not contain the hidden checks. A task may include visible tests in `repo/` as hints. The hidden checks in `verify/` are what decide the result.

## 5. Runner (`bench/run.py`)

**Usage:**

```
py bench/run.py [--tasks a,b] [--repeat N] [--budget N] [--xolotl PATH] [--note TEXT] [--keep]
```

Defaults: all tasks, `--repeat 1`, Xolotl's own budget default, build from the source tree.

**Steps:**

1. **Build.** Unless `--xolotl` is given, run `cargo build --release`, then copy the binary to a temp location. The copy keeps a later rebuild from hitting a locked `.exe`.
2. **Record the run metadata:** git SHA, a dirty flag from `git status --porcelain`, the model, and `xolotl --version`.
3. **For each task, for each attempt:**
   1. Copy `repo/` into a fresh temp directory. Run `git init` and an initial commit there (author `bench <bench@local>`), so the agent has git available.
   2. Run `xolotl -p <prompt> --mode yolo --cwd <tmp> [--budget N]`:
      - capture stdout (JSON-lines events) and stderr;
      - stop it at `timeout_s` by killing the whole process tree (`taskkill /T /F /PID` on Windows, the process group elsewhere);
      - record the exit code and wall time.
   3. Copy `verify/` into `<tmp>/verify/`, overwriting anything the agent created at that path. Run the `verify` commands and record pass/fail plus the last 2 KB of their output.
   4. Compute metrics from the captured events (§6). Find the session folder as the newest `~/.xolotl/projects/<slug>/sessions/*`, using the same slug rule as `store::project_slug`, and record its path.
   5. Delete the temp directory unless `--keep` is set or the task failed. A failed task's directory is kept for inspection and its path is recorded.
4. **Write** `results/<run_id>.json` and append a row to `results/HISTORY.md`.
5. **Print** a summary table. The runner exits 0 when the run completes, whatever the pass rate.

**Rules:**

- The API key never passes through the runner. Xolotl reads it itself.
- The runner never prints environment variables.
- Tasks run sequentially.
- Ctrl+C stops the run cleanly: it kills the current child tree, writes a partial results file marked `"partial": true`, and does not append to `HISTORY.md`.

## 6. Metrics and results

**Metrics per attempt,** all derived from the event stream:

| Metric | Source |
|---|---|
| `pass` | every verify command exited 0 |
| `exit_code`, `timed_out`, `wall_s` | runner |
| `model_calls` | count of `assistant` events |
| `tool_calls` | count of `tool_result` events |
| `tokens.prompt` / `completion` / `cached` | sum of `assistant.usage` fields |
| `compactions.stage1` / `stage2` | count of `compaction` events by stage |
| `completion_check` | whether a `nudge` with reason `completion_check` occurred |
| `errors` | count of `error` events |
| `policy_hash` | from the `session_start` event |

**`results/<run_id>.json`:**

```json
{
  "run_id": "20261004-1530-82ad7b2", "git_sha": "82ad7b2…", "dirty": false,
  "xolotl_version": "xolotl 0.1.0", "model": "glm-5.3-flash", "policy_hash": "…",
  "repeat": 3, "budget": null, "note": "v1 baseline", "partial": false,
  "attempts": [ { "task": "py-offbyone", "attempt": 1, "pass": true, "…": "…" } ],
  "summary": {
    "pass": 31, "total": 36, "pass_rate": 0.861,
    "per_task": { "py-offbyone": "3/3", "…": "…" },
    "tokens": { "prompt": 0, "completion": 0, "cached": 0 },
    "median_wall_s": 0, "total_wall_s": 0
  }
}
```

**`results/HISTORY.md`** is a table with these columns: date · run_id · policy_hash (first 8 characters) · pass (k/N and %) · prompt / completion tokens · median wall time · note.

## 7. The 12 tasks

Each task is small and deterministic to check. Each exercises a different skill.

| id | Kind | What the agent must do | Hidden check |
|---|---|---|---|
| `py-offbyone` | Python bugfix | fix a moving-average off-by-one; a failing visible test is the hint | extended unit tests |
| `py-docstring` | Python implement | implement `slugify` from its docstring | unit tests, including edge cases |
| `py-rename` | Python refactor | rename `get_usr` to `get_user` across a 4-file package | no `get_usr` left, and the package tests pass |
| `py-argparse` | Python feature | add `--limit N` to a CLI script | runs the CLI with and without the flag |
| `py-logparse` | Python data | write `summarize.py` that turns `access.log` into `summary.json` | compares against the expected JSON |
| `py-crossmodule` | Python bugfix | a failing order test whose bug lives in `pricing.py` | extended unit tests |
| `py-perf` | Python performance | make an O(n²) `dedupe` handle 200k items in under 2 s, keeping order | correctness plus timing |
| `rs-compile` | Rust | make `cargo test` pass in a small crate with a type error and a borrow error, without editing the tests | `cargo test`, and the test file's hash is unchanged |
| `rs-method` | Rust feature | add `Stack::peek` with a test | hidden test file plus `cargo test` |
| `js-fix` | Node bugfix | fix `formatDuration` | node assertion script |
| `env-tool` | Environment | write a script that prints the SHA-256 of each `data/*.txt`, sorted by name | compares the output |
| `long-context` | Long context and compaction | across 25 small modules, list every function that writes to the global `CONFIG` in `FINDINGS.md`; runs with `budget: 15000` | exact set comparison |

Toolchains used: Python 3, Rust/cargo, Node and Git Bash. All four are present on this machine. A task whose toolchain is missing is reported as `skipped` and doesn't count as a failure.

## 8. Baseline

- After the suite lands, run `py bench/run.py --repeat 3 --note "v1 baseline"` on the v1 code, plus this branch's `bench/` changes.
- Commit the results file and the `HISTORY.md` row.
- That row is the reference point for every later change.

## 9. Testing the runner

`bench/run.py` gets a small standard-library `unittest` file, `bench/test_run.py`, covering:

- metrics extraction from a canned event stream;
- the slug function matching Rust's `project_slug` cases (`D:\Xolotl Harness` becomes `D--Xolotl-Harness`);
- `HISTORY.md` row formatting;
- `task.json` validation (missing fields are rejected).

Each task also has a self-check. Running its `verify` against a reference solution kept in `tasks/<id>/solution/` must pass, and running it against the untouched `repo/` must fail. The runner exposes this as `--selfcheck`, which proves every task is solvable and not already solved, with no model calls.

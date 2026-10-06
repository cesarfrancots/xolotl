# Xolotl bench

Runs Xolotl headless on a suite of small coding tasks, checks each result with hidden tests, and records pass rate,
tokens and time so later versions can be compared with the v1 baseline. Python standard library only; Windows 11 with
Git Bash is the primary platform. Run everything from the repo root.

## Run

```
py bench/run.py --selfcheck                          # no model calls: every task must be solvable and not already solved
py bench/run.py --repeat 3 --note "v1 baseline"      # full suite, 3 attempts per task (builds with cargo first)
py bench/run.py --tasks py-offbyone --keep           # one task; --keep leaves each attempt's temp dir behind
py bench/run.py --xolotl path\to\xolotl.exe          # use a ready binary instead of building (a .py file works too)
py bench/run.py --fingerprint                        # print each task's hash and the suite hash; no build, no model calls
py -m unittest bench.test_run -v                     # tests for the runner itself
```

Other options: `--budget N` (passed to Xolotl as `--budget`; a task's own `budget` wins), `--tasks-dir DIR`.
Xolotl reads the API key itself; the runner never reads or prints environment variables.

Each run writes `results/<run_id>.json` (`run_id` = `YYYYMMDD-HHMM-<sha7>`, with `-2`, `-3` appended if that file
already exists, so a run never overwrites another) and appends one row to `results/HISTORY.md`. Ctrl+C, or an error
in the runner itself, stops the run, writes the finished attempts with `"partial": true` (plus an `error` message for
an error), and adds no history row. `dirty` is the state of the source tree when the run started, ignoring
`bench/results/`; a dirty run's HISTORY row says `(dirty)` after its run id. A failed attempt's temp directory is kept
and its path recorded (`kept_dir`); the hidden `verify/` is removed from it. The session trace stays in `~/.xolotl/`
(`session_dir`).

Each results file also records what the run was made of, so two runs can be compared like for like: `git_sha`,
`policy_hash` (from Xolotl's `session_start`), `tasks_hash` (see below), `xolotl_version`, `xolotl_source` (`built`, or
the `--xolotl` path) and `xolotl_config` (the parsed `~/.xolotl/config.json`, or null if there is none; it holds no
secret, since Xolotl rejects unknown fields and reads the key from an environment variable).

`policy_hash` hashes the bytes of `src/policy.rs`, so it, and every comparison across checkouts, assumes LF line
endings. The repo's `.gitattributes` (`* text=auto eol=lf`) makes Windows checkouts use LF despite `core.autocrlf`.

`tasks_hash` fingerprints the suite that ran: a SHA-256 (12 hex) over each selected task's `task_hash`, which covers
`task.json`, `repo/` and `verify/` (paths and bytes, CRLF read as LF; not `solution/`). It's in each attempt (`task_hash`),
in each run, and in the HISTORY `Tasks` column; if it differs between two runs, the tasks changed and the pass rates
aren't comparable. `--fingerprint` prints the same hashes without running anything.

## Add a task

```
tasks/<id>/task.json   definition (below)
tasks/<id>/repo/       files the agent starts with (visible tests may be hints, but never the hidden checks)
tasks/<id>/verify/     hidden checks, copied in as <tmp>/verify only after the agent finishes
tasks/<id>/solution/   reference solution, overlaid on repo/ by --selfcheck only
```

`task.json` fields: `id` (must equal the folder name), `title`, `prompt`, `verify` (list of shell commands, all must
exit 0), and optionally `tags`, `timeout_s` (600), `verify_timeout_s` (120), `budget` (null), `requires` (executables
the task needs; if one is missing the task is `skipped`, not failed).

`{python}` in a verify command is replaced by the runner's own interpreter. Commands run in the task's temp directory
through Git Bash. The agent can't see `verify/` or `solution/`, and any `verify/` it creates is deleted before the real one
is copied in. Prefer expected values computed from the inputs inside `verify/` over hard-coded ones. Finish with
`--selfcheck`.

## Metrics

Per attempt, from Xolotl's JSON-lines events: `pass` (every verify command exited 0), `exit_code`, `timed_out`,
`wall_s` (Xolotl's run only), `model_calls` (`assistant` events), `tool_calls` (`tool_result` events), `tokens`
(`prompt` / `completion` / `cached`, summed from `assistant.usage`), `compactions` (`stage1` / `stage2` counts),
`completion_check` (a `completion_check` nudge happened), `errors` (`error` events), `policy_hash` and `model` (from
`session_start`), `verify_output` and `stderr_tail` (last 2 KB of the checks' output, last 1 KB of Xolotl's stderr).
`status` is `pass`, `fail`, `skipped` or `infra_error`.

- `infra_error` means the attempt says nothing about the agent: no `session_start` event at all (for example the API key
  variable is not set), or an `error` event of kind `llm` that isn't retried (quota, auth, network, HTTP). `infra_reason`
  says which; `stderr_tail` usually says why. A context-overflow error doesn't count (the agent compacts and retries, or
  fails by its own context management), and an attempt whose checks passed stays `pass`.
- `skipped` and `infra_error` attempts count in `total` but not in the pass rate, `per_task` denominators or wall-time
  figures. A task's `per_task` entry reads like `2/2 (+1 infra)`; `summary.infra_errors` counts them.
- The run still gets its HISTORY row, with the pass cell flagged, for example `2/3 (66.7%) ⚠1 infra`, and the runner
  prints a warning. Re-run the affected tasks before trusting such a row.

`summary` totals these per run: pass count and rate, `per_task` as `k/N`, summed tokens, median and total wall time.
The HISTORY columns are: Date, Run, Policy (first 8 of `policy_hash`), Tasks (`tasks_hash`), Pass, Tokens, Median wall, Note.

## Task notes

- `long-context` (`budget: 15000`): compaction depends on the agent's read strategy. Reading the files one by one usually crosses
  stage 1 (60% of the budget); grepping first may never compact, so `compactions` can legitimately be 0 on a passing attempt.
- `long-context`: every `mod_NN:name` token in `FINDINGS.md` counts, so listing non-writers in that format (for example in a
  "not writers" section) fails the check.
- `env-tool`: each line must use the bare file name, not `data/<name>`, and Git Bash's `sha256sum` default ` *name` marker fails
  the check. Both are stated in the prompt ("<sha256 hex>  <file name>", two spaces).

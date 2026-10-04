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
py -m unittest bench.test_run -v                     # tests for the runner itself
```

Other options: `--budget N` (passed to Xolotl as `--budget`; a task's own `budget` wins), `--tasks-dir DIR`.
Xolotl reads the API key itself; the runner never reads or prints environment variables.

Each run writes `results/<run_id>.json` (`run_id` = `YYYYMMDD-HHMM-<sha7>`, with `-2`, `-3` appended if that file
already exists, so a run never overwrites another) and appends one row to `results/HISTORY.md`. Ctrl+C, or an error
in the runner itself, stops the run, writes the finished attempts with `"partial": true` (plus an `error` message for
an error), and adds no history row. `dirty` is the state of the source tree when the run started, ignoring
`bench/results/`. A failed attempt's temp directory is kept and its path recorded (`kept_dir`); the hidden `verify/`
is removed from it. The session trace stays in `~/.xolotl/` (`session_dir`).

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
`status` is `pass`, `fail` or `skipped`; skipped attempts count in `total` but not in `pass_rate`.

`summary` totals these per run: pass count and rate, `per_task` as `k/N`, summed tokens, median and total wall time.

## Task notes

- `long-context` (`budget: 15000`): compaction depends on the agent's read strategy. Reading the files one by one usually crosses
  stage 1 (60% of the budget); grepping first may never compact, so `compactions` can legitimately be 0 on a passing attempt.
- `long-context`: every `mod_NN:name` token in `FINDINGS.md` counts, so listing non-writers in that format (for example in a
  "not writers" section) fails the check.
- `env-tool`: each line must use the bare file name, not `data/<name>`, and Git Bash's `sha256sum` default ` *name` marker fails
  the check. Both are stated in the prompt ("<sha256 hex>  <file name>", two spaces).

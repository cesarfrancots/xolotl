# Xolotl on Terminal-Bench 2.0 (via Harbor)

Runs Xolotl headless (`xolotl -p ... --mode yolo`) inside each Terminal-Bench 2.0 (TB2) task container with the
[Harbor](https://github.com/harbor-framework/harbor) framework, and records pass rate, tokens and agent time in the same
tracked form as the local suite (`bench/`). Python standard library only for our code; Harbor is a separate tool.
Windows 11 + Git Bash is the primary platform. Run everything from the repo root.

## Prerequisites

- Docker Desktop with the Linux engine (TB2 images are linux/amd64). Each task image is about 0.5 GB or more and is
  pulled on first use, then cached (the smoke pair is about 1.1 GB).
- [uv](https://docs.astral.sh/uv/) and Harbor 0.23.x: `uv tool install harbor` (uv fetches a Python >= 3.12 for it).
  `harbor --version` must work from the shell that runs the wrapper.
- `ZAI_API_KEY` in the process environment or the Windows User environment (read with `reg query`, as Xolotl does).

## Run

```
py bench/tb2/run.py --smoke --note "tb2 smoke"        # the 2 smoke tasks (smoke.txt)
py bench/tb2/run.py --note "tb2 v1 baseline"          # the fixed 15-task subset (tasks.txt)
py bench/tb2/run.py --task fix-git --task regex-log   # any TB2 tasks by name
py bench/tb2/run.py --tasks-file bench/tb2/tasks-hard.txt -n 2 --note "..."   # the 12 unseen hard tasks (see Task sets)
py bench/tb2/run.py --tasks-file my.txt               # your own list: one name per line, # comments
py bench/tb2/run.py --smoke --dry-run                 # print the Harbor command (key redacted), let Harbor validate task names; no run, no build
                                                      # (still needs the Docker daemon up, but pulls and runs nothing; it silently drops
                                                      # unknown task names, so check that "N trial(s)" equals your task count)
py bench/tb2/run.py --reparse bench/tb2/jobs/<run_id> --note "..."   # re-derive results from a finished job folder (see Results)
py -m unittest bench.tb2.test_tb2 -v                  # tests for the pure helpers
```

Options: `-n N` Harbor concurrency (default 1, to keep Docker load low), `-k N` attempts per task (default 1; one
trial per task swings about ±2 tasks run to run on the hard set, so compare changes at `-k 3`), `--no-build` reuse the built binary,
`--jobs-dir`, `--results-dir`. The wrapper builds the static musl binary first (`build_linux.py`: `rust:1.95-alpine` in
Docker with `cargo build --release --locked`, cargo registry cached in the `xolotl-cargo-registry` volume, then checks the
ELF has no dynamic loader), runs
`harbor run -d terminal-bench@2.0 -a xolotl_harbor.agent:XolotlAgent -m zai/glm-5.3-flash --ak binary=<path> ...`, parses
the job, scans it for the key, and writes the results.

## How the agent is found

`harbor` runs in its own uv-managed interpreter, so the custom agent must be importable there. The wrapper does it with
`PYTHONPATH=bench/tb2` in Harbor's environment (so `xolotl_harbor.agent:XolotlAgent` resolves); `--with-editable` is not
needed. To call Harbor by hand, set the same variable, e.g. in Git Bash:

```
PYTHONPATH=bench/tb2 harbor run -d terminal-bench@2.0 -a xolotl_harbor.agent:XolotlAgent -m zai/glm-5.3-flash \
  --ak binary=target/x86_64-unknown-linux-musl/release/xol -i fix-git
```

`XolotlAgent.install()` uploads the binary to `/installed-agent/xolotl` (kwarg `binary`, else env `XOLOTL_LINUX_BIN`);
`run()` executes one shell command whose instruction is base64-encoded, so quoting can't break it. Xolotl runs under
`timeout -s INT -k 5 <task timeout - 60 s>` (SIGINT, which Xolotl handles like Ctrl+C so the call in flight is saved to the
trace, then SIGKILL 5 s later, so it is guaranteed gone; floor 60 s; see Timeouts) and `--deadline <same seconds>`, so it
lowers its reasoning effort, adds time notes as the budget runs down, and cuts a thought that has reasoned for `max(150 s, half the time left)` without a tool call or an answer (`policy.rs`). The command always exits 0: Xolotl's own exit code goes
to `/logs/agent/exit_code`, and the verifier decides pass or fail. `core.py` copies the metrics and infra functions of
`bench/run.py` (Harbor's interpreter can't import `bench`); a test asserts identical output.

## Timeouts

Tasks differ (TB2 has 600 s to 12000 s agent timeouts; the hard set below mixes 900 s and 1800 s), and Harbor doesn't pass
the agent its task's timeout. `XolotlAgent.run()` therefore finds it itself, per trial, with no wrapper involvement
(`core.resolve_timeout`), in this order:

1. `--ak agent_timeout_sec=N`, if given: an explicit override (use it with Harbor's own `--agent-timeout-multiplier` or
   `--agent-timeout`, which the agent can't see). A bad value (0, negative, not a number) fails when the agent is created.
2. The task's own `[agent].timeout_sec`: Harbor gives `run()` an environment whose `environment_dir` is the task's
   `<task>/environment` folder (Harbor's `trial.py`, `environment_dir=self.task.paths.environment_dir`), and `task.toml`
   sits next to it (for the registry dataset: `~/.cache/harbor/tasks/<hash>/<task>/task.toml`).
3. Otherwise 900 s, recorded as source `default`.

Xolotl's budget is that value minus 60 s (floor 60 s), so it stops before Harbor kills the container and the logs are
still copied: 840 s for a 900 s task, 1740 s for a 1800 s one. What each trial really got is written to
`<trial>/agent/agent_timeout.json` (`agent_timeout_sec`, `source` = `kwarg` / `task.toml` / `default`, `budget_s`) and ends
up in each trial of `results/<run_id>.json` as `agent_timeout_sec` and `timeout_source` (null for a job from before this
existed or a trial that never reached `run()`), and in the `limit` column of the summary. A `default` trial prints a
WARNING: it means `task.toml` wasn't found and the budget may be shorter than the task's real limit. The earlier results
(smoke and the 15-task subset) only ran 900 s tasks, so their fixed 840 s was right and they stay comparable.

## Results

Each run writes `results/<run_id>.json` (`run_id` = `YYYYMMDD-HHMM-<sha7>`, also the Harbor job name; the raw job is in
`jobs/<run_id>/`, git-ignored) and appends a row to `results/HISTORY.md`:
`Date | Run | Policy | Tasks | Pass | Tokens (in / out) | Median agent | Note`. `Tasks` is the sha256 (12 hex) of the sorted
task names, so two rows are comparable only when it matches.

A trial's `exit_code` is Xolotl's own: 0 done, 2 step limit, 124 `timeout` stopped it at its time budget (the verifier
still grades whatever it left behind).

Per trial: `status` is `pass` (verifier reward >= 1), `fail`, or `infra_error`. The verifier decides: reward >= 1 is a
pass even with an agent timeout or a non-zero Xolotl exit code, and reward < 1 is a fail, unless `xolotl.jsonl` has no
`session_start` or an unretried `llm` error (the agent never really ran: infra). With no reward at all, a verifier-phase
exception (`VerifierTimeoutError`, `RewardFileNotFoundError`, `RewardFileEmptyError`, `VerifierOutputParseError`) is a
fail when a real Xolotl session ran; any other exception (setup, install, Docker, `AddTestsDirError`,
`DownloadVerifierDirError`...) is infra. Infra trials say nothing about the agent: they are excluded from the pass rate
and the row shows `⚠N infra`. A measured pass stays a pass. Tokens and metadata are derived from `xolotl.jsonl` (Harbor's
`agent_result` is only the fallback if that file is missing). Each trial also records Harbor's `task_checksum`, and the
run records `binary_sha256` (the binary uploaded into every container). A partial run (Harbor failed or fewer trials
than tasks) is written with `"partial": true` and gets no HISTORY row.

Re-parsing: `py bench/tb2/run.py --reparse bench/tb2/jobs/<run_id> [--note TEXT]` re-derives the results of an existing
job folder with the current parser (no build, no Harbor, no Docker): it runs the secrets scan on the folder, writes
`results/<run_id>.json` (`-2`, `-3` appended if that file exists) with `"reparsed": true` and the parser's git sha
(`parser_git_sha`), and appends the HISTORY row. Provenance the job folder doesn't hold stays unknown: `git_sha` is only the
sha7 in the job name, `dirty` and `harbor_version` are null, and `binary_sha256` is that of the binary on disk now (the
run's own only if it wasn't rebuilt since). The job folder is only read.

## Task sets

`tasks.txt` (15 tasks: 3 easy, 6 medium, 6 hard, 900 s timeouts), `smoke.txt` (2 easy) and `tasks-hard.txt` are fixed;
changing them changes `tasks_hash`.

`tasks-hard.txt` (selected 2026-10-05) is 12 TB2 tasks of `difficulty = "hard"` that are not in `tasks.txt`, no GPU, spread
over categories (selection rule at the top of the file):

| Agent timeout (Xolotl budget) | Tasks |
|---|---|
| 900 s (840 s) | gpt2-codegolf, make-doom-for-mips, polyglot-rust-c, torch-pipeline-parallelism, torch-tensor-parallelism, write-compressor |
| 1800 s (1740 s) | feal-differential-cryptanalysis, make-mips-interpreter, path-tracing, protein-assembly, mcmc-sampling-stan, llm-inference-batching-scheduler |

Resource notes: mcmc-sampling-stan asks for 4 cpus / 8 GB, the torch tasks, gpt2-codegolf and protein-assembly for 4 GB,
the rest 1 cpu / 2 GB, so keep Docker Desktop's memory limit at 8 GB or more and use `-n 2` at most. Image sizes were not
screened (the torch images carry PyTorch, gpt2-codegolf bakes in the GPT-2 124M checkpoint), so the first run pulls a lot.
The 1800 s tasks are only fair because of the Timeouts section above: before it, Xolotl was killed at 840 s.

## Secrets

The key travels only in the Harbor child's environment and into the container through `exec_as_agent(env=...)`; it is
never in our argv, `--ae`, agent kwargs, Harbor's config, logs, files or commits. It does appear in the host
`docker compose exec` command line that Harbor runs during a trial (visible to local processes of the same user), which
is outside our control. After every run the wrapper scans `jobs/<run_id>/` for the key's first 8 characters (it prints
paths only); on a hit it exits 3 and writes no results. A file or folder the scan cannot read counts as a failed scan
too (warning line per path, exit 3, no results; the job folder is left intact, and `--reparse` repeats the scan once the
access problem is fixed).

## Fallback

If Harbor cannot run on native Windows (path or CRLF bugs, harbor-framework/harbor issues #1239 and #1240), the fallback is
WSL Ubuntu with Docker Desktop's WSL integration, running the same commands there. That is a user decision.

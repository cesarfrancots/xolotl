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
py bench/tb2/run.py --tasks-file my.txt               # your own list: one name per line, # comments
py bench/tb2/run.py --smoke --dry-run                 # print the Harbor command (key redacted), let Harbor validate task names; no run, no build
py -m unittest bench.tb2.test_tb2 -v                  # tests for the pure helpers
```

Options: `-n N` Harbor concurrency (default 1, to keep Docker load low), `--no-build` reuse the built binary,
`--jobs-dir`, `--results-dir`. The wrapper builds the static musl binary first (`build_linux.py`: `rust:1.95-alpine` in
Docker, cargo registry cached in the `xolotl-cargo-registry` volume, then checks the ELF has no dynamic loader), runs
`harbor run -d terminal-bench@2.0 -a xolotl_harbor.agent:XolotlAgent -m zai/glm-5.3-flash --ak binary=<path> ...`, parses
the job, scans it for the key, and writes the results.

## How the agent is found

`harbor` runs in its own uv-managed interpreter, so the custom agent must be importable there. The wrapper does it with
`PYTHONPATH=bench/tb2` in Harbor's environment (so `xolotl_harbor.agent:XolotlAgent` resolves); `--with-editable` is not
needed. To call Harbor by hand, set the same variable, e.g. in Git Bash:

```
PYTHONPATH=bench/tb2 harbor run -d terminal-bench@2.0 -a xolotl_harbor.agent:XolotlAgent -m zai/glm-5.3-flash \
  --ak binary=target/x86_64-unknown-linux-musl/release/xolotl -i fix-git
```

`XolotlAgent.install()` uploads the binary to `/installed-agent/xolotl` (kwarg `binary`, else env `XOLOTL_LINUX_BIN`);
`run()` executes one shell command whose instruction is base64-encoded, so quoting can't break it. Xolotl runs under
`timeout <task timeout - 60 s>` (default 840 s; Harbor doesn't tell an agent its task's timeout, so pass
`--ak agent_timeout_sec=N` for a task whose timeout isn't 900 s). The command always exits 0: Xolotl's own exit code goes
to `/logs/agent/exit_code`, and the verifier decides pass or fail. `core.py` copies the metrics and infra functions of
`bench/run.py` (Harbor's interpreter can't import `bench`); a test asserts identical output.

## Results

Each run writes `results/<run_id>.json` (`run_id` = `YYYYMMDD-HHMM-<sha7>`, also the Harbor job name; the raw job is in
`jobs/<run_id>/`, git-ignored) and appends a row to `results/HISTORY.md`:
`Date | Run | Policy | Tasks | Pass | Tokens (in / out) | Median agent | Note`. `Tasks` is the sha256 (12 hex) of the sorted
task names, so two rows are comparable only when it matches.

A trial's `exit_code` is Xolotl's own: 0 done, 2 step limit, 124 `timeout` stopped it at its time budget (the verifier
still grades whatever it left behind).

Per trial: `status` is `pass` (verifier reward >= 1), `fail`, or `infra_error`. A trial with no verifier result (setup,
environment or verifier exception), or whose `xolotl.jsonl` has no `session_start` or an unretried `llm` error, says
nothing about the agent; it is excluded from the pass rate and the row shows `⚠N infra`. A measured pass stays a pass.
A partial run (Harbor failed or fewer trials than tasks) is written with `"partial": true` and gets no HISTORY row.

`tasks.txt` (15 tasks: 3 easy, 6 medium, 6 hard, 900 s timeouts) and `smoke.txt` (2 easy) are fixed; changing them changes
`tasks_hash`.

## Secrets

The key travels only in the Harbor child's environment and into the container through `exec_as_agent(env=...)`; never in
argv, `--ae`, kwargs, files, logs or commits. After every run the wrapper scans `jobs/<run_id>/` for the key's first 8
characters (it prints paths only); on a hit it exits 3 and writes no results.

## Fallback

If Harbor cannot run on native Windows (path or CRLF bugs, harbor-framework/harbor issues #1239 and #1240), the fallback is
WSL Ubuntu with Docker Desktop's WSL integration, running the same commands there. That is a user decision.

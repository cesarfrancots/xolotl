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
py bench/tb2/run.py --reparse bench/tb2/jobs/<run_id> --note "..."   # re-derive results from a finished job folder (see Results)
py -m unittest bench.tb2.test_tb2 -v                  # tests for the pure helpers
```

Options: `-n N` Harbor concurrency (default 1, to keep Docker load low), `--no-build` reuse the built binary,
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
`timeout -k 5 <task timeout - 60 s>` (SIGKILL 5 s after SIGTERM, so it is guaranteed gone; default 840 s; Harbor doesn't
tell an agent its task's timeout, so pass `--ak agent_timeout_sec=N` for a task whose timeout isn't 900 s). The command
always exits 0: Xolotl's own exit code goes
to `/logs/agent/exit_code`, and the verifier decides pass or fail. `core.py` copies the metrics and infra functions of
`bench/run.py` (Harbor's interpreter can't import `bench`); a test asserts identical output.

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

`tasks.txt` (15 tasks: 3 easy, 6 medium, 6 hard, 900 s timeouts) and `smoke.txt` (2 easy) are fixed; changing them changes
`tasks_hash`.

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

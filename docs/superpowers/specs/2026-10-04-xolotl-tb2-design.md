# Xolotl Bench — Terminal-Bench 2.0 Adapter Design Spec (sub-project 2)

- **Date:** 2026-10-04
- **Status:** Approved in conversation (design, host = native Windows first, scope = smoke then a ~15-task subset)
- **Depends on:**
  - Xolotl v1 (PR #1)
  - the local bench (PR #2), for its results/HISTORY conventions
- **Facts as of:** Harbor 0.23.0, researched 2026-10-04. Sources are in §10.

## 1. Purpose

Run Xolotl on Terminal-Bench 2.0 (TB2), the benchmark the Meta-Harness paper reports, through the Harbor framework.

Each run records results in the same tracked form as the local suite. The local suite already saturates at 36/36, so TB2 is what can show capability changes between versions.

## 2. Goals and non-goals

**Goals**

1. A Harbor custom agent (`XolotlAgent`, an installed agent) that runs Xolotl inside each TB2 task container.
2. A reproducible static Linux build of Xolotl (x86_64 musl), made inside Docker.
3. A wrapper that:
   - builds,
   - runs Harbor on the smoke set or the fixed subset,
   - collects per-task results,
   - writes `bench/tb2/results/<run_id>.json`,
   - appends a `bench/tb2/results/HISTORY.md` row.
4. A committed TB2 baseline on a fixed ~15-task subset, run once per task.
5. The API key never appears in any file, job folder, log or commit.

**Non-goals**

- The full 89-task run (later, quota permitting).
- Cloud sandboxes (Modal/Daytona).
- Running Harbor inside WSL. That is only the fallback if native Windows fails.
- ATIF trajectory export.
- Any change to Xolotl's own behavior. The adapter only drives the existing headless mode.

## 3. Host setup

| Requirement | Detail |
|---|---|
| Docker Desktop | Linux engine. TB2 images are linux/amd64 only. |
| Harbor | `uv tool install harbor` (Python ≥ 3.12). The adapter is made importable with `--with-editable bench/tb2` or `PYTHONPATH`; pick whichever works and document it in the README. |
| Disk | TB2 images are large, so expect several GB of image pulls. |

If Harbor fails on native Windows in a way that blocks the smoke run (path or CRLF bugs, as in harbor-framework/harbor issues #1239 and #1240), stop and report. The documented fallback is WSL Ubuntu with Docker Desktop's WSL integration; the user installs that.

## 4. Layout

```
bench/tb2/
  xolotl_harbor/__init__.py
  xolotl_harbor/agent.py      XolotlAgent(BaseInstalledAgent) + pure helpers
  build_linux.py              static musl build via `docker run rust:<ver>-alpine`
  run.py                      wrapper (smoke / subset / custom), results + HISTORY
  test_tb2.py                 unittest for the pure helpers (fixtures under fixtures/)
  tasks.txt                   fixed subset, one TB2 task name per line, with a comment header
  smoke.txt                   the 2 smoke tasks
  results/<run_id>.json, results/HISTORY.md
  jobs/                       Harbor job output (git-ignored)
  README.md
```

## 5. The agent (`XolotlAgent`)

### 5.1 Identity

- `name()` returns `"xolotl"`.
- The version command is `/installed-agent/xolotl --version`.

### 5.2 `install(environment)`

- Upload the host musl binary to `/installed-agent/xolotl` and `chmod +x` it.
- The binary path comes from agent kwarg `binary=` (`--ak binary=<path>`) or the host env var `XOLOTL_LINUX_BIN`.
- If the binary is missing on the host, fail fast with a clear error.

### 5.3 `run(instruction, environment, context)`

Execute one command through `exec_as_agent`:

```
cd "$PWD"   # the task's own WORKDIR (not always /app)
timeout <T> /installed-agent/xolotl -p "$(printf %s '<b64>' | base64 -d)" --mode yolo --cwd "$PWD" --model <model> \
  > /logs/agent/xolotl.jsonl 2> /logs/agent/xolotl.stderr < /dev/null
echo $? > /logs/agent/exit_code
mkdir -p /logs/agent/xolotl-home && cp -r /root/.xolotl/. /logs/agent/xolotl-home/ 2>/dev/null
true
```

**The instruction.** It is base64-encoded on the host, so no quoting can break or inject.

**The timeout `T`.**
- `T` = the task's agent timeout minus 60 s, floored at 60 s.
- The task timeout comes from the task config if Harbor exposes it; otherwise use a 900 s default.
- `timeout` makes sure Xolotl stops before Harbor's own deadline, and the logs still get copied.

**The model.**
- `<model>` = Harbor's `-m` value with any `provider/` prefix stripped, e.g. `zai/glm-5.3-flash` becomes `glm-5.3-flash`.
- Default `glm-5.3-flash`.

**The key.** `env={"ZAI_API_KEY": <key>}`, where `<key>` comes from the host process environment via `_get_env`. It is never placed in `--ae` and never in kwargs.

**Exit status.** The command always exits 0. Xolotl's real exit code is in `/logs/agent/exit_code`. An agent exit code therefore does not become a Harbor exception, and the verifier decides pass/fail as usual.

### 5.4 Metrics (`populate_context_post_run`, or the end of `run`)

Parse `<trial>/agent/xolotl.jsonl` on the host. Set:

| Context field | Value |
|---|---|
| `n_input_tokens` | Σ prompt |
| `n_cache_tokens` | Σ cached |
| `n_output_tokens` | Σ completion |
| `metadata` | `{model_calls, tool_calls, compactions{stage1,stage2}, policy_hash, exit_code, infra_reason}` |

`infra_reason` uses the same classification as the local suite's `bench/run.py`. Reuse it by importing `bench.run` helpers, or by copying the small pure function if importing is awkward. Either way, behaviour must be identical.

## 6. Build (`build_linux.py`)

Run this from the host:

```
docker run --rm -v <repo>:/src -w /src rust:1.95-alpine \
  sh -c "apk add musl-dev && cargo build --release --target x86_64-unknown-linux-musl"
```

Output: `target/x86_64-unknown-linux-musl/release/xolotl`.

- The binary must be static. Verify with `file`, or by checking that `ldd` reports "not a dynamic executable" inside the build container.
- The build prints the binary path and its policy hash.
- Cargo's registry is cached in a named Docker volume so rebuilds are fast.

## 7. Wrapper (`run.py`)

**Usage:**

```
py bench/tb2/run.py [--smoke | --tasks-file F | --task NAME ...] [-n CONCURRENCY=2] [--note TEXT] [--keep-jobs] [--no-build]
```

**Steps:**

1. Build (unless `--no-build`). Check that Docker is running; if not, exit with a clear message.
2. Read the key into the Harbor child's environment.
   - Source: process env, else the Windows User scope via `reg query`, as `xolotl` itself does.
   - The key is never printed and never placed on a command line.
3. Run Harbor:
   ```
   harbor run -d terminal-bench@2.0 -a xolotl_harbor.agent:XolotlAgent -m zai/glm-5.3-flash \
     --ak binary=<path> -n <N> -o bench/tb2/jobs --job-name <run_id> -i <task> [-i <task> …]
   ```
   - Use the include flag once per task, if Harbor allows repeating it; otherwise a glob or one job per task.
   - Verify the exact task-name form with `--dry-run` first.
4. Parse `jobs/<run_id>/result.json` and each trial's `result.json`:
   - reward (pass = reward ≥ 1.0, or the task's own pass semantics if reward.json carries one),
   - exceptions,
   - agent context metrics,
   - wall time.
5. **Infra classification.** A trial is `infra_error` if any of these hold:
   - Harbor recorded an exception from setup/build/environment rather than from the verifier;
   - `xolotl.jsonl` has no `session_start`;
   - an `llm` error with retry false that isn't a context overflow occurred.

   An infra trial is excluded from the pass rate and shown as `⚠N infra`, as in the local suite.
6. Write `results/<run_id>.json` and append a HISTORY row:

   | Column | Content |
   |---|---|
   | Date | |
   | Run | `<run_id>` |
   | Policy | `policy_hash` |
   | Tasks | `tasks_hash` (sha256 of the sorted task-name list, 12 hex) |
   | Pass | k/N, % and infra |
   | Tokens | input / output |
   | Median agent time | |
   | Note | |

   `run_id = YYYYMMDD-HHMM-<sha7>`. Use exclusive create with a `-2`/`-3` suffix on collision, as the local suite does.
7. Unless `--keep-jobs`, keep the raw jobs folder anyway; it is git-ignored. Print its path, and print a summary table.

**Secrets scan.** After every run, scan `jobs/<run_id>/` for the key's first 8 characters without printing the key. If anything matches, refuse to write results and exit nonzero.

## 8. Task sets

**`smoke.txt`:** 2 quick, low-dependency TB2 tasks with 900 s timeouts (chosen from the TB2 task list, e.g. `fix-git`).

**`tasks.txt`:** 15 tasks.
- About 5 easy, 5 medium and 5 hard, by TB2's own `difficulty` metadata.
- Varied categories: software engineering, data, security, sysadmin, scientific.
- Prefer 900 s agent timeouts to keep a run at about 1.5–3 h at `-n 2`.
- The header comment records the selection rule and the date.

The subset is fixed once committed. Changing it changes `tasks_hash`.

## 9. Testing

**`bench/tb2/test_tb2.py`** (unittest) covers these pure helpers:

- the run-command builder: base64 round-trip of a nasty instruction (quotes, `$()`, newlines, unicode), `timeout` computation and floor, and model-prefix stripping;
- the jsonl → metrics parser: reuses the local suite's fixtures and tolerates garbage lines;
- the Harbor-result parser: fixture `result.json` files for a pass, a fail, a verifier pass with an agent exit code of 2, and an infra exception;
- the `tasks_hash` and HISTORY row format;
- the secrets scan on a fixture dir: a planted fake key is detected.

**Integration:** the `--smoke` run against the live API. It must finish with both tasks graded (pass or fail, not infra), tokens populated, and the secrets scan clean.

## 10. Sources

- Harbor custom agents: https://docs.harborframework.com/agents/custom-agents.md
- Harbor install: https://docs.harborframework.com/getting-started/installation.md
- Harbor full docs (env vars, datasets): https://docs.harborframework.com/llms-full.txt
- Harbor repo (`agents/installed/base.py`, `environments/docker/docker.py`, `cli/jobs.py`): https://github.com/harbor-framework/harbor
- TB2 tasks: https://github.com/harbor-framework/terminal-bench-2 and https://huggingface.co/datasets/harborframework/terminal-bench-2.0
- Windows issues: https://github.com/harbor-framework/harbor/issues/1240
- Community custom-agent example: https://github.com/badlogic/pi-terminal-bench

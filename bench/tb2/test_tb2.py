import asyncio
import base64
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from bench import run as local_run
from bench.tb2 import run
from bench.tb2.xolotl_harbor import core

HERE = Path(__file__).parent
JOB = HERE / "fixtures" / "job"
EDGE = HERE / "fixtures" / "job_edge"
FAKE_KEY = "FAKEKEY1.not-a-real-secret"


def ev(type_: str, **fields) -> str:
    # ensure_ascii=False like serde_json: U+2028, U+2029 and U+0085 stay raw inside the line
    return json.dumps({"type": type_, **fields}, ensure_ascii=False)


def assistant(prompt: int, completion: int, cached: int) -> str:
    return ev("assistant", text="x", usage={"prompt": prompt, "completion": completion, "cached": cached})


STREAM = [
    ev("session_start", cwd="x", model="m", mode="yolo", policy_hash="abc", system=""),
    assistant(100, 20, 50), ev("tool_result", id="1", output="ok"),
    assistant(200, 30, 0),
]

# A task.toml shaped like TB2's (the layout Harbor caches: <task>/task.toml next to <task>/environment/)
TASK_TOML = """version = "1.0"

[metadata]
difficulty = "hard"

[verifier]
timeout_sec = 1200.0

[agent]
timeout_sec = {timeout}

[environment]
build_timeout_sec = 600.0
cpus = 1
"""


def make_task(root: Path, timeout: str | None = "1800.0", toml: str | None = None) -> Path:
    """<root>/task/{task.toml, environment/}; returns the environment dir, which is what Harbor hands the agent."""
    env = root / "task" / "environment"
    env.mkdir(parents=True)
    if timeout is not None or toml is not None:
        (root / "task" / "task.toml").write_text(toml if toml is not None else TASK_TOML.format(timeout=timeout), encoding="utf-8")
    return env


class CoreTests(unittest.TestCase):
    def test_model_arg(self):
        self.assertEqual(core.model_arg("zai/glm-5.3-flash"), "glm-5.3-flash")
        self.assertEqual(core.model_arg("glm-5.3-flash"), "glm-5.3-flash")
        self.assertEqual(core.model_arg(None), "glm-5.3-flash")

    def test_timeout_budget(self):
        self.assertEqual(core.timeout_budget(900), 840)
        self.assertEqual(core.timeout_budget(60), 60)
        self.assertEqual(core.timeout_budget(30), 60)
        self.assertEqual(core.timeout_budget(None), 840)
        self.assertEqual(core.timeout_budget(1800.0), 1740)  # task.toml holds floats; the shell wants an int

    def test_task_agent_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(core.task_agent_timeout(make_task(Path(d))), 1800.0)
        for timeout in ("900", "900.5"):
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(core.task_agent_timeout(make_task(Path(d), timeout)), float(timeout))
        # no task.toml, a garbled one, or no usable [agent].timeout_sec: None, never an exception
        bad = {"missing file": None, "not toml": "{{{", "no [agent]": "[verifier]\ntimeout_sec = 5\n",
               "no timeout_sec": "[agent]\nuser = 'root'\n", "agent is a number": "agent = 3\n",
               "zero": TASK_TOML.format(timeout="0"), "negative": TASK_TOML.format(timeout="-60"),
               "string": TASK_TOML.format(timeout='"900"'), "bool": TASK_TOML.format(timeout="true")}
        for label, toml in bad.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as d:
                self.assertIsNone(core.task_agent_timeout(make_task(Path(d), None, toml)))
        self.assertIsNone(core.task_agent_timeout(Path("/does/not/exist/environment")))

    def test_resolve_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            env = make_task(Path(d))
            self.assertEqual(core.resolve_timeout(None, env), {"agent_timeout_sec": 1800.0, "source": "task.toml", "budget_s": 1740})
            # an explicit --ak agent_timeout_sec beats the task's own value
            self.assertEqual(core.resolve_timeout(600, env), {"agent_timeout_sec": 600.0, "source": "kwarg", "budget_s": 540})
            self.assertEqual(core.resolve_timeout(30, env)["budget_s"], 60)  # the 60 s floor still applies
            for bad in (0, -5, "abc", True, float("nan")):  # a typo'd override fails loudly, never silently
                with self.subTest(bad), self.assertRaises(ValueError):
                    core.resolve_timeout(bad, env)
        with tempfile.TemporaryDirectory() as d:  # no task.toml found: the old default, but labelled so it can be flagged
            self.assertEqual(core.resolve_timeout(None, make_task(Path(d), None)),
                             {"agent_timeout_sec": 900, "source": "default", "budget_s": 840})
        self.assertEqual(core.resolve_timeout(None, None)["source"], "default")

    def test_command_base64_roundtrip(self):
        instruction = 'say "hi"; $(rm -rf /) `x`\nline2 — ünïcode'
        cmd = core.run_command(instruction, "glm-5.3-flash", 840)
        self.assertNotIn("rm -rf", cmd)  # no raw instruction text reaches the shell
        self.assertNotIn("line2", cmd)
        literals = re.findall(r"printf %s '([A-Za-z0-9+/=]+)'", cmd)
        self.assertEqual(len(literals), 1)
        self.assertEqual(base64.b64decode(literals[0]).decode("utf-8"), instruction)
        for part in ("timeout -s INT -k 5 840 /installed-agent/xolotl", "--mode yolo", '--cwd "$PWD"', "--model glm-5.3-flash",
                     "--deadline 840 > /logs/agent/xolotl.jsonl", "echo $? > /logs/agent/exit_code", "< /dev/null"):
            self.assertIn(part, cmd)
        self.assertTrue(cmd.rstrip().endswith("true"))

    def test_command_quotes_model(self):
        cmd = core.run_command("hi", "evil model; rm -rf /", 60)
        self.assertIn("--model 'evil model; rm -rf /' --deadline 60 >", cmd)  # one shell word, whatever -m contained

    def test_metrics_from_jsonl(self):
        m = core.metrics_from_jsonl("\n".join(STREAM))
        self.assertEqual(m["tokens"], {"prompt": 300, "completion": 50, "cached": 50})
        self.assertEqual((m["model_calls"], m["tool_calls"]), (2, 1))
        self.assertEqual(m["policy_hash"], "abc")
        self.assertIsNone(m["infra_reason"])
        self.assertEqual(core.metrics_from_jsonl("")["infra_reason"], "no session_start event")

    def test_metrics_match_local_suite(self):
        """core copies bench.run's metrics/infra functions (Harbor's interpreter can't import `bench`); same output."""
        streams = [
            STREAM,
            [],
            ["", "not json", '{"type":"assistant"', "[1,2]", "null"] + STREAM,
            STREAM + [ev("compaction", stage=1), ev("compaction", stage=2), ev("nudge", reason="completion_check"),
                      ev("error", kind="net", message="boom", retry=True), ev("interrupt", during="tool")],
            STREAM + [ev("error", kind="llm", message="quota exhausted", retry=False)],
            STREAM + [ev("error", kind="llm", message="context window exceeded: 9 > 8", retry=False)],
            STREAM + [ev("error", kind="llm", message="slow", retry=True)],
            [ev("assistant", usage={"prompt": "7", "completion": True, "cached": -1})],
            # serde_json leaves U+2028, U+2029 and U+0085 unescaped inside strings; str.splitlines() would cut there
            [ev("session_start", cwd="x", model="m", mode="yolo", policy_hash="abc", system="a b\u0085c d"),
             ev("assistant", text="line one\u0085two end", usage={"prompt": 100, "completion": 1, "cached": 2})],
        ]
        for lines in streams:
            expected = local_run.metrics_from_events(lines)
            events = list(local_run._events(lines))
            reason = local_run._infra_reason(events)
            got = core.metrics_from_jsonl("\n".join(lines))
            self.assertEqual(got.pop("infra_reason"), reason or None, lines)
            self.assertEqual(got, expected, lines)

    def test_unicode_line_separators_do_not_drop_events(self):
        stream = [ev("session_start", cwd="x", model="m", mode="yolo", policy_hash="abc", system=""),
                  ev("assistant", text="a b\u0085c d", usage={"prompt": 100, "completion": 1, "cached": 0})]
        m = core.metrics_from_jsonl("\n".join(stream) + "\n")
        self.assertEqual((m["model_calls"], m["tokens"]["prompt"], m["policy_hash"], m["infra_reason"]), (1, 100, "abc", None))

    def test_read_api_key_prefers_process_env(self):
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": "k-from-env"}):
            self.assertEqual(core.read_api_key(), "k-from-env")


class AgentTests(unittest.TestCase):
    """agent.py imports Harbor, which only exists in its own uv interpreter: stub the three modules it imports so the
    real XolotlAgent.run() is exercised under plain `py`."""

    @contextlib.contextmanager
    def agent_module(self):
        class Base:  # the slice of BaseInstalledAgent that XolotlAgent touches
            def __init__(self, logs_dir=None, model_name=None, **kwargs):
                self.logs_dir, self.model_name, self.logger, self.calls = Path(logs_dir), model_name, mock.Mock(), []

            async def exec_as_agent(self, environment, cmd, env=None):
                self.calls.append((cmd, env))

        stubs = {}
        for name, attr in (("harbor.agents.installed.base", "BaseInstalledAgent"),
                           ("harbor.environments.base", "BaseEnvironment"), ("harbor.models.agent.context", "AgentContext")):
            stubs[name] = types.ModuleType(name)
            setattr(stubs[name], attr, Base if attr == "BaseInstalledAgent" else object)
        with mock.patch.dict(sys.modules, stubs):  # on exit sys.modules is restored, so the stub-bound module isn't kept
            yield importlib.import_module("bench.tb2.xolotl_harbor.agent")

    def run_agent(self, env_dir: Path | None, logs: Path, **kwargs):
        with self.agent_module() as mod, mock.patch.object(core, "read_api_key", return_value="fake-key"):
            agent = mod.XolotlAgent(logs_dir=logs, model_name="zai/glm-5.3-flash", **kwargs)
            asyncio.run(agent.run("do it", types.SimpleNamespace(environment_dir=env_dir), None))
        return agent.calls

    def test_each_task_gets_its_own_timeout(self):
        for task_timeout, budget in (("900.0", 840), ("1800.0", 1740)):
            with self.subTest(task_timeout), tempfile.TemporaryDirectory() as d:
                logs = Path(d) / "agent"
                logs.mkdir()
                (cmd, env), = self.run_agent(make_task(Path(d), task_timeout), logs)
                self.assertIn(f"timeout -s INT -k 5 {budget} /installed-agent/xolotl", cmd)
                self.assertIn(f" --deadline {budget} > ", cmd)  # Xolotl paces itself by the same budget it is killed at
                self.assertEqual(env, {"ZAI_API_KEY": "fake-key"})  # the key only via env=, never in the command
                self.assertNotIn("fake-key", cmd)
                rec = json.loads((logs / "agent_timeout.json").read_text(encoding="utf-8"))
                self.assertEqual(rec, {"agent_timeout_sec": float(task_timeout), "source": "task.toml", "budget_s": budget})

    def test_override_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / "agent"
            logs.mkdir()
            (cmd, _), = self.run_agent(make_task(Path(d)), logs, agent_timeout_sec=600)  # --ak agent_timeout_sec=600
            self.assertIn("timeout -s INT -k 5 540 ", cmd)
            self.assertIn(" --deadline 540 ", cmd)
            self.assertEqual(json.loads((logs / "agent_timeout.json").read_text(encoding="utf-8"))["source"], "kwarg")
        with tempfile.TemporaryDirectory() as d:  # no task.toml reachable: the default, recorded as such
            logs = Path(d) / "agent"
            logs.mkdir()
            (cmd, _), = self.run_agent(make_task(Path(d), None), logs)
            self.assertIn("timeout -s INT -k 5 840 ", cmd)
            self.assertEqual(json.loads((logs / "agent_timeout.json").read_text(encoding="utf-8"))["source"], "default")

    def test_bad_override_fails_at_construction(self):
        with self.agent_module() as mod, self.assertRaises(ValueError):
            mod.XolotlAgent(logs_dir=Path("x"), model_name="zai/m", agent_timeout_sec="soon")

    def test_unwritable_record_does_not_stop_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            calls = self.run_agent(make_task(Path(d)), Path(d) / "no-such-logs-dir")  # the record can't be written
            self.assertEqual(len(calls), 1)
            self.assertIn("timeout -s INT -k 5 1740 ", calls[0][0])


class WrapperTests(unittest.TestCase):
    def test_parse_job_statuses(self):
        trials = {t["task"]: t for t in run.parse_job(JOB)}
        self.assertEqual({k: v["status"] for k, v in trials.items()}, {
            "t-pass": "pass", "t-fail": "fail",
            "t-exit2": "pass",  # reward 1 with Xolotl's exit code 2 (step limit): the verifier decides
            "t-setup": "infra_error", "t-nosession": "infra_error"})
        self.assertEqual(trials["t-exit2"]["exit_code"], 2)
        self.assertEqual(trials["t-setup"]["infra_reason"], "RuntimeError")
        self.assertEqual(trials["t-nosession"]["infra_reason"], "no session_start event")
        self.assertEqual(trials["t-pass"]["tokens"], {"prompt": 300, "completion": 50, "cached": 50})
        self.assertEqual(trials["t-pass"]["agent_time_s"], 100.0)
        self.assertEqual(trials["t-pass"]["policy_hash"], "abc12345def")
        self.assertEqual(trials["t-pass"]["xolotl_version"], "xolotl 0.1.0")
        # a trial that never ran the agent (no agent files, no timing) still parses
        self.assertIsNone(trials["t-setup"]["agent_time_s"])
        self.assertEqual(trials["t-setup"]["tokens"], {"prompt": 0, "completion": 0, "cached": 0})

    def test_parse_trial_records_agent_timeout(self):
        trials = {t["task"]: t for t in run.parse_job(JOB)}
        self.assertEqual((trials["t-pass"]["agent_timeout_sec"], trials["t-pass"]["timeout_source"]), (1800.0, "task.toml"))
        self.assertEqual((trials["t-fail"]["agent_timeout_sec"], trials["t-fail"]["timeout_source"]), (900.0, "task.toml"))
        # no record (a job from before it existed, or a trial that never reached run()): unknown, not a guess
        self.assertEqual((trials["t-exit2"]["agent_timeout_sec"], trials["t-exit2"]["timeout_source"]), (None, None))
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t-pass__a1"
            shutil.copytree(JOB / "t-pass__a1", t)
            for junk in ("not json", "[1]", '{"agent_timeout_sec": "soon", "source": 7}', '{"agent_timeout_sec": true}'):
                (t / "agent" / "agent_timeout.json").write_text(junk, encoding="utf-8")
                got = run.parse_trial(t)
                self.assertEqual((got["agent_timeout_sec"], got["timeout_source"]), (None, None), junk)

    def test_summary_flags_default_timeout(self):
        trial = {"task": "t", "status": "pass", "reward": 1.0, "agent_time_s": 5.0, "infra_reason": "", "exit_code": 0,
                 "tokens": {"prompt": 1, "completion": 1, "cached": 0}, "agent_timeout_sec": 900, "timeout_source": "default"}
        result = {"trials": [trial], "summary": run.summarize([trial])}
        for source, flagged in (("default", True), ("task.toml", False), ("kwarg", False), (None, False)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run._print_summary(result | {"trials": [trial | {"timeout_source": source}]}, Path("r.json"))
            self.assertEqual("task.toml was not found" in out.getvalue(), flagged, source)

    def test_summarize_excludes_infra(self):
        s = run.summarize(run.parse_job(JOB))
        self.assertEqual((s["pass"], s["total"], s["infra"]), (2, 5, 2))
        self.assertAlmostEqual(s["pass_rate"], 2 / 3, places=3)
        # tokens come from each xolotl.jsonl (300/50/50 for the 3 trials with a real session; t-nosession has none),
        # not from the fixtures' agent_result, which says 300/50/50 for t-nosession too
        self.assertEqual(s["tokens"], {"prompt": 900, "completion": 150, "cached": 150})
        self.assertEqual(s["median_agent_time_s"], 200.0)  # pass 100 s, fail 200 s, exit2 300 s; infra excluded

    def test_parse_job_edge_cases(self):
        trials = {t["task"]: t for t in run.parse_job(EDGE)}
        self.assertEqual({k: v["status"] for k, v in trials.items()}, {
            "t-timeout-fail": "fail",  # AgentTimeoutError, reward 0: the agent ran out of time
            "t-timeout-pass": "pass",  # AgentTimeoutError, reward 1: the verifier decides
            "t-verifier-timeout": "fail",  # verifier exception behind a real session is the agent's failure
            "t-verifier-nosession": "infra_error",  # ... but not if the agent never really ran
            "t-addtests": "infra_error",  # AddTestsDirError is not a verifier outcome
            "t-separators": "pass"})
        self.assertEqual(trials["t-timeout-fail"]["exception"], "AgentTimeoutError")
        self.assertEqual(trials["t-timeout-pass"]["exception"], "AgentTimeoutError")
        self.assertEqual(trials["t-timeout-fail"]["infra_reason"], "")
        self.assertIsNone(trials["t-verifier-timeout"]["reward"])
        self.assertEqual(trials["t-verifier-nosession"]["infra_reason"], "no session_start event")
        self.assertEqual(trials["t-addtests"]["infra_reason"], "AddTestsDirError")
        # events with raw U+2028 / U+0085 / U+2029 inside a string are all counted
        sep = trials["t-separators"]
        self.assertEqual(sep["tokens"], {"prompt": 300, "completion": 50, "cached": 50})
        self.assertEqual((sep["metadata"]["model_calls"], sep["metadata"]["tool_calls"]), (2, 1))
        self.assertEqual(sep["infra_reason"], "")
        self.assertEqual(sep["task_checksum"], "checksum-of-t-separators")

    def test_tokens_prefer_jsonl_over_harbor_context(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t-pass__a1"
            shutil.copytree(JOB / "t-pass__a1", t)
            r = json.loads((t / "result.json").read_text(encoding="utf-8"))
            r["agent_result"].update(n_input_tokens=1, n_output_tokens=2, n_cache_tokens=3)  # as a buggy parse would
            (t / "result.json").write_text(json.dumps(r), encoding="utf-8")
            self.assertEqual(run.parse_trial(t)["tokens"], {"prompt": 300, "completion": 50, "cached": 50})
            (t / "agent" / "xolotl.jsonl").unlink()  # no jsonl: Harbor's context is the fallback
            self.assertEqual(run.parse_trial(t)["tokens"], {"prompt": 1, "completion": 2, "cached": 3})

    def test_history_row_format(self):
        result = {"run_id": "20261004-1200-abc1234", "started": "2026-10-04T12:00:00+02:00", "dirty": False,
                  "policy_hash": "e381398fdeadbeef", "tasks_hash": "c7bd70e76232", "note": "tb2 | smoke",
                  "summary": run.summarize(run.parse_job(JOB))}
        self.assertEqual(
            run.history_row(result),
            "| 2026-10-04 12:00 | 20261004-1200-abc1234 | e381398f | c7bd70e76232 | 2/3 (66.7%) ⚠2 infra | "
            "900 / 150 | 200.0s | tb2 \\| smoke |")
        self.assertIn("(dirty)", run.history_row({**result, "dirty": True}))
        self.assertTrue(run.HISTORY_HEADER.startswith("| Date | Run | Policy | Tasks | Pass | Tokens (in / out) | Median agent | Note |"))

    def test_tasks_hash_order_independent(self):
        self.assertEqual(run.tasks_hash(["b", "a", "c"]), run.tasks_hash(["c", "a", "b"]))
        self.assertNotEqual(run.tasks_hash(["a", "b"]), run.tasks_hash(["a", "c"]))
        self.assertRegex(run.tasks_hash(["a"]), r"^[0-9a-f]{12}$")

    def test_secret_scan_detects_planted_key(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub").mkdir()
            (root / "clean.txt").write_text("nothing here", encoding="utf-8")
            self.assertEqual(run.secret_scan(root, FAKE_KEY), ([], []))
            (root / "sub" / "log.txt").write_text(f"env dump: ZAI_API_KEY={FAKE_KEY[:8]}...", encoding="utf-8")
            self.assertEqual(run.secret_scan(root, FAKE_KEY), ([root / "sub" / "log.txt"], []))

    def test_secret_scan_unreadable_file_is_not_clean(self):
        real_read = Path.read_bytes

        def read_bytes(self):
            if self.name == "locked.txt":
                raise PermissionError("denied")
            return real_read(self)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ok.txt").write_text("nothing here", encoding="utf-8")
            (root / "locked.txt").write_text("could hold anything", encoding="utf-8")
            with mock.patch.object(Path, "read_bytes", read_bytes):
                self.assertEqual(run.secret_scan(root, FAKE_KEY), ([], [root / "locked.txt"]))
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    self.assertEqual(run._scan_failed(root, FAKE_KEY), 3)
            self.assertIn(f"WARNING: cannot read {root / 'locked.txt'}", err.getvalue())
            self.assertIn("is intact", err.getvalue())
            self.assertEqual(run._scan_failed(root, FAKE_KEY), 0)  # readable again: clean

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bin"
            p.write_bytes(b"x" * 3_000_000)
            self.assertEqual(run.file_sha256(p), hashlib.sha256(b"x" * 3_000_000).hexdigest())

    def _reparse(self, job: Path, results: Path, note: str = "") -> int:
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": FAKE_KEY}), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return run.main(["--reparse", str(job), "--results-dir", str(results), "--note", note])

    def test_reparse_writes_results_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            job, results = Path(d) / "20261004-1200-abc1234", Path(d) / "results"
            shutil.copytree(JOB, job)
            before = {p.relative_to(job).as_posix(): p.read_bytes() for p in job.rglob("*") if p.is_file()}
            self.assertEqual(self._reparse(job, results, "re|parse"), 0)
            data = json.loads((results / "20261004-1200-abc1234.json").read_text(encoding="utf-8"))
            self.assertIs(data["reparsed"], True)
            self.assertRegex(data["parser_git_sha"], r"^([0-9a-f]{40}|unknown)$")
            self.assertEqual(data["git_sha"], "abc1234")
            self.assertEqual((data["summary"]["pass"], data["summary"]["total"], data["summary"]["infra"]), (2, 5, 2))
            self.assertEqual(data["tasks_hash"], run.tasks_hash(["t-pass", "t-fail", "t-exit2", "t-setup", "t-nosession"]))
            self.assertEqual({t["task"]: t["task_checksum"] for t in data["trials"]}["t-pass"], "checksum-of-t-pass")
            self.assertEqual({t["task"]: t["agent_timeout_sec"] for t in data["trials"]},
                             {"t-pass": 1800.0, "t-fail": 900.0, "t-exit2": None, "t-setup": None, "t-nosession": None})
            self.assertIn("binary_sha256", data)
            rows = (results / "HISTORY.md").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)  # header, separator, one row
            self.assertIn("20261004-1200-abc1234 | abc12345 | ", rows[2])
            self.assertIn("2/3 (66.7%) ⚠2 infra", rows[2])
            self.assertTrue(rows[2].endswith("re\\|parse |"))
            # a second reparse never overwrites the first result: run_id gets a suffix
            self.assertEqual(self._reparse(job, results), 0)
            self.assertEqual(json.loads((results / "20261004-1200-abc1234-2.json").read_text(encoding="utf-8"))["run_id"],
                             "20261004-1200-abc1234-2")
            self.assertEqual(len((results / "HISTORY.md").read_text(encoding="utf-8").splitlines()), 4)
            # and the job folder is only ever read
            self.assertEqual(before, {p.relative_to(job).as_posix(): p.read_bytes() for p in job.rglob("*") if p.is_file()})

    def test_reparse_refuses_when_key_prefix_in_job(self):
        with tempfile.TemporaryDirectory() as d:
            job, results = Path(d) / "job", Path(d) / "results"
            shutil.copytree(JOB, job)
            (job / "t-pass__a1" / "agent" / "leak.txt").write_text(f"ZAI_API_KEY={FAKE_KEY[:8]}", encoding="utf-8")
            self.assertEqual(self._reparse(job, results), 3)
            self.assertFalse(results.exists())  # nothing written, job folder untouched

    def test_task_lists(self):
        smoke, subset = run.read_task_names(HERE / "smoke.txt"), run.read_task_names(HERE / "tasks.txt")
        self.assertEqual((len(smoke), len(subset)), (2, 15))
        self.assertEqual(len(set(subset)), 15)
        self.assertTrue(all(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", t) for t in smoke + subset))

    def test_hard_task_list(self):
        hard = run.read_task_names(HERE / "tasks-hard.txt")
        self.assertEqual(len(hard), 12)
        self.assertEqual(len(set(hard)), 12)
        self.assertTrue(all(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", t) for t in hard))
        self.assertFalse(set(hard) & set(run.read_task_names(HERE / "tasks.txt")))  # unseen: none already in the fixed subset

    def test_key_not_in_our_argv(self):  # Harbor itself puts it in its `docker compose exec` argv while a trial runs
        cmd = run.harbor_command("harbor", ["a", "b"], Path("/x/xolotl"), Path("/jobs"), "job1", 1)
        self.assertEqual(cmd[cmd.index("-n") + 1], "1")
        self.assertEqual([cmd[i + 1] for i, c in enumerate(cmd) if c == "-i"], ["a", "b"])
        env = run.harbor_env("sekrit-key")
        self.assertEqual(env[core.KEY_ENV], "sekrit-key")
        self.assertFalse(any("sekrit-key" in part for part in cmd))
        self.assertIn(str(HERE), env["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()

import base64
import json
import re
import tempfile
import unittest
from pathlib import Path

from bench import run as local_run
from bench.tb2 import run
from bench.tb2.xolotl_harbor import core

HERE = Path(__file__).parent
JOB = HERE / "fixtures" / "job"


def ev(type_: str, **fields) -> str:
    return json.dumps({"type": type_, **fields})


def assistant(prompt: int, completion: int, cached: int) -> str:
    return ev("assistant", text="x", usage={"prompt": prompt, "completion": completion, "cached": cached})


STREAM = [
    ev("session_start", cwd="x", model="m", mode="yolo", policy_hash="abc", system=""),
    assistant(100, 20, 50), ev("tool_result", id="1", output="ok"),
    assistant(200, 30, 0),
]


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

    def test_command_base64_roundtrip(self):
        instruction = 'say "hi"; $(rm -rf /) `x`\nline2 — ünïcode'
        cmd = core.run_command(instruction, "glm-5.3-flash", 840)
        self.assertNotIn("rm -rf", cmd)  # no raw instruction text reaches the shell
        self.assertNotIn("line2", cmd)
        literals = re.findall(r"printf %s '([A-Za-z0-9+/=]+)'", cmd)
        self.assertEqual(len(literals), 1)
        self.assertEqual(base64.b64decode(literals[0]).decode("utf-8"), instruction)
        for part in ("timeout 840 /installed-agent/xolotl", "--mode yolo", '--cwd "$PWD"', "--model glm-5.3-flash",
                     "> /logs/agent/xolotl.jsonl", "echo $? > /logs/agent/exit_code", "< /dev/null"):
            self.assertIn(part, cmd)
        self.assertTrue(cmd.rstrip().endswith("true"))

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
        ]
        for lines in streams:
            expected = local_run.metrics_from_events(lines)
            events = list(local_run._events(lines))
            reason = local_run._infra_reason(events)
            got = core.metrics_from_jsonl("\n".join(lines))
            self.assertEqual(got.pop("infra_reason"), reason or None, lines)
            self.assertEqual(got, expected, lines)

    def test_read_api_key_prefers_process_env(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": "k-from-env"}):
            self.assertEqual(core.read_api_key(), "k-from-env")


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

    def test_summarize_excludes_infra(self):
        s = run.summarize(run.parse_job(JOB))
        self.assertEqual((s["pass"], s["total"], s["infra"]), (2, 5, 2))
        self.assertAlmostEqual(s["pass_rate"], 2 / 3, places=3)
        self.assertEqual(s["tokens"], {"prompt": 1200, "completion": 200, "cached": 200})  # 4 trials ran the agent
        self.assertEqual(s["median_agent_time_s"], 200.0)  # pass 100 s, fail 200 s, exit2 300 s; infra excluded

    def test_history_row_format(self):
        result = {"run_id": "20261004-1200-abc1234", "started": "2026-10-04T12:00:00+02:00", "dirty": False,
                  "policy_hash": "e381398fdeadbeef", "tasks_hash": "c7bd70e76232", "note": "tb2 | smoke",
                  "summary": run.summarize(run.parse_job(JOB))}
        self.assertEqual(
            run.history_row(result),
            "| 2026-10-04 12:00 | 20261004-1200-abc1234 | e381398f | c7bd70e76232 | 2/3 (66.7%) ⚠2 infra | "
            "1200 / 200 | 200.0s | tb2 \\| smoke |")
        self.assertIn("(dirty)", run.history_row({**result, "dirty": True}))
        self.assertTrue(run.HISTORY_HEADER.startswith("| Date | Run | Policy | Tasks | Pass | Tokens (in / out) | Median agent | Note |"))

    def test_tasks_hash_order_independent(self):
        self.assertEqual(run.tasks_hash(["b", "a", "c"]), run.tasks_hash(["c", "a", "b"]))
        self.assertNotEqual(run.tasks_hash(["a", "b"]), run.tasks_hash(["a", "c"]))
        self.assertRegex(run.tasks_hash(["a"]), r"^[0-9a-f]{12}$")

    def test_secret_scan_detects_planted_key(self):
        key = "FAKEKEY1.not-a-real-secret"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub").mkdir()
            (root / "clean.txt").write_text("nothing here", encoding="utf-8")
            self.assertEqual(run.secret_scan(root, key), [])
            (root / "sub" / "log.txt").write_text(f"env dump: ZAI_API_KEY={key[:8]}...", encoding="utf-8")
            self.assertEqual(run.secret_scan(root, key), [root / "sub" / "log.txt"])

    def test_task_lists(self):
        smoke, subset = run.read_task_names(HERE / "smoke.txt"), run.read_task_names(HERE / "tasks.txt")
        self.assertEqual((len(smoke), len(subset)), (2, 15))
        self.assertEqual(len(set(subset)), 15)
        self.assertTrue(all(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", t) for t in smoke + subset))

    def test_key_never_in_argv(self):
        cmd = run.harbor_command("harbor", ["a", "b"], Path("/x/xolotl"), Path("/jobs"), "job1", 1)
        self.assertEqual(cmd[cmd.index("-n") + 1], "1")
        self.assertEqual([cmd[i + 1] for i, c in enumerate(cmd) if c == "-i"], ["a", "b"])
        env = run.harbor_env("sekrit-key")
        self.assertEqual(env[core.KEY_ENV], "sekrit-key")
        self.assertFalse(any("sekrit-key" in part for part in cmd))
        self.assertIn(str(HERE), env["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()

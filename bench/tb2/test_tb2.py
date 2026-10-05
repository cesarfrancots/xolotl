import base64
import json
import re
import unittest

from bench import run as local_run
from bench.tb2.xolotl_harbor import core


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


if __name__ == "__main__":
    unittest.main()

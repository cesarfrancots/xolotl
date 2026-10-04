import json
import tempfile
import unittest
from pathlib import Path

from bench import run


def write_task(root: Path, folder: str, **fields) -> Path:
    """Create root/folder/task.json holding `fields` and return the task dir."""
    d = root / folder
    d.mkdir()
    (d / "task.json").write_text(json.dumps(fields), encoding="utf-8")
    return d


def ev(type_: str, **fields) -> str:
    return json.dumps({"seq": 0, "t_ms": 0, "type": type_, **fields})


def assistant(prompt: int = 0, completion: int = 0, cached: int = 0) -> str:
    return ev("assistant", text="", reasoning="", tool_calls=[],
              usage={"prompt": prompt, "completion": completion, "cached": cached},
              finish_reason="stop", latency_ms=1)


def tool_result() -> str:
    return ev("tool_result", call_id="c", name="bash", ok=True, output="", bytes=0, duration_ms=1)


class Pure(unittest.TestCase):
    def test_slug_matches_rust(self):
        self.assertEqual(run.slug(r"D:\Xolotl Harness"), "D--Xolotl-Harness")
        self.assertEqual(run.slug("/home/me/my proj"), "-home-me-my-proj")
        self.assertEqual(run.slug("a_b-c.d"), "a_b-c.d")  # kept as-is
        self.assertEqual(run.slug("café"), "caf-")  # one '-' per non-ASCII char

    def test_load_task_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {"id": "good", "title": "T", "prompt": "do it", "verify": ["true"]}

            t = run.load_task(write_task(root, "good", **base))
            self.assertEqual(t["id"], "good")
            self.assertEqual(t["tags"], [])
            self.assertEqual(t["timeout_s"], 600)
            self.assertEqual(t["verify_timeout_s"], 120)
            self.assertIsNone(t["budget"])
            self.assertEqual(t["requires"], [])

            full = {**base, "id": "full", "tags": ["py"], "timeout_s": 30, "verify_timeout_s": 5,
                    "budget": 30000, "requires": ["cargo"]}
            t = run.load_task(write_task(root, "full", **full))
            self.assertEqual((t["tags"], t["timeout_s"], t["verify_timeout_s"], t["budget"], t["requires"]),
                             (["py"], 30, 5, 30000, ["cargo"]))

            # defaults are fresh per call, not shared mutable state
            run.load_task(root / "good")["tags"].append("x")
            self.assertEqual(run.load_task(root / "good")["tags"], [])

            def rejects(folder: str, needle: str, **fields):
                with self.subTest(folder=folder):
                    with self.assertRaises(ValueError) as cm:
                        run.load_task(write_task(root, folder, **fields))
                    self.assertIn(needle, str(cm.exception))

            no_prompt = {k: v for k, v in base.items() if k != "prompt"}
            rejects("noprompt", "prompt", **{**no_prompt, "id": "noprompt"})
            rejects("emptyverify", "verify", **{**base, "id": "emptyverify", "verify": []})
            rejects("blankverify", "verify", **{**base, "id": "blankverify", "verify": [" "]})  # would pass vacuously
            rejects("strverify", "verify", **{**base, "id": "strverify", "verify": "true"})
            rejects("other-name", "id", **base)  # id "good" != folder "other-name"
            rejects("badtimeout", "timeout_s", **{**base, "id": "badtimeout", "timeout_s": "10"})
            rejects("zerotimeout", "timeout_s", **{**base, "id": "zerotimeout", "timeout_s": 0})
            rejects("badbudget", "budget", **{**base, "id": "badbudget", "budget": "1k"})
            rejects("badreq", "requires", **{**base, "id": "badreq", "requires": "cargo"})

            with self.assertRaises(ValueError):  # no task.json at all
                run.load_task(root / "nothing-here")
            broken = root / "broken"
            broken.mkdir()
            (broken / "task.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                run.load_task(broken)

    def test_metrics_from_events(self):
        lines = [
            ev("session_start", cwd="x", model="m", mode="yolo", policy_hash="abc", system=""),
            assistant(100, 20, 50), tool_result(),
            assistant(200, 30, 0), tool_result(),
            ev("compaction", stage=1), ev("compaction", stage=2),
            ev("nudge", reason="completion_check", text="done?"),
            ev("error", kind="net", message="boom", retry=True),
            ev("interrupt", during="tool"),
        ]
        self.assertEqual(run.metrics_from_events(lines), {
            "model_calls": 2, "tool_calls": 2,
            "tokens": {"prompt": 300, "completion": 50, "cached": 50},
            "compactions": {"stage1": 1, "stage2": 1},
            "completion_check": True, "errors": 1, "policy_hash": "abc", "interrupted": True,
        })

        quiet = run.metrics_from_events([ev("nudge", reason="repeat", text=""), ev("session_end", status="ok")])
        self.assertFalse(quiet["completion_check"])
        self.assertFalse(quiet["interrupted"])
        self.assertEqual(quiet["policy_hash"], "")
        self.assertEqual(quiet["model_calls"], 0)

    def test_metrics_tolerate_garbage_lines(self):
        lines = ["", "not json", '{"type":"assistant"', b"\xff".decode("latin-1"), "[1,2]", "7", "null",
                 "[" * 200_000,  # deep nesting raises RecursionError, not ValueError
                 ev("assistant", usage=None), assistant(5, 6, 7),
                 ev("compaction", stage=1.0), ev("compaction", stage=[1]), ev("compaction", stage=3)]
        m = run.metrics_from_events(lines)
        self.assertEqual(m["model_calls"], 2)
        self.assertEqual(m["tokens"], {"prompt": 5, "completion": 6, "cached": 7})
        self.assertEqual(m["compactions"], {"stage1": 0, "stage2": 0})  # odd stage values are ignored, not a crash
        self.assertEqual(run.metrics_from_events(iter([])), run.metrics_from_events([]))  # any iterable

    def test_session_dir_picks_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cwd = r"D:\Xolotl Harness\tmp 1"
            self.assertIsNone(run.session_dir(home, cwd))  # no project dir at all

            sessions = home / ".xolotl" / "projects" / "D--Xolotl-Harness-tmp-1" / "sessions"
            sessions.mkdir(parents=True)
            self.assertIsNone(run.session_dir(home, cwd))  # project dir, no sessions yet

            (sessions / "20261004-110000-bbbb").mkdir()
            (sessions / "20261004-100000-aaaa").mkdir()
            (sessions / "20261004-120000-stray.txt").write_text("not a session")
            self.assertEqual(run.session_dir(home, cwd), sessions / "20261004-110000-bbbb")

    def test_summarize_and_history_row(self):
        def att(task, status, p=0, c=0, wall=0.0):
            return {"task": task, "attempt": 1, "status": status, "pass": status == "pass",
                    "tokens": {"prompt": p, "completion": c, "cached": 0}, "wall_s": wall}

        attempts = [att("t1", "pass", 100, 20, 10.0), att("t1", "fail", 100, 20, 12.0),
                    att("t2", "pass", 100, 10, 14.0), {"task": "t3", "attempt": 1, "status": "skipped", "pass": False}]
        s = run.summarize(attempts)
        self.assertEqual((s["pass"], s["total"], s["skipped"], s["pass_rate"]), (2, 4, 1, 0.667))
        self.assertEqual(s["per_task"], {"t1": "1/2", "t2": "1/1", "t3": "skipped"})
        self.assertEqual(s["tokens"], {"prompt": 300, "completion": 50, "cached": 0})
        self.assertEqual((s["median_wall_s"], s["total_wall_s"]), (12.0, 36.0))

        all_skipped = run.summarize([{"task": "t", "attempt": 1, "status": "skipped", "pass": False}])
        self.assertEqual(all_skipped["pass_rate"], 0.0)  # no division by zero
        self.assertEqual(run.summarize([])["pass_rate"], 0.0)

        result = {"run_id": "20261004-1530-82ad7b2", "started": "2026-10-04T15:30:12",
                  "policy_hash": "abcdef1234567890", "note": "v1 baseline", "summary": s}
        row = run.history_row(result)
        self.assertEqual(row, "| 2026-10-04 15:30 | 20261004-1530-82ad7b2 | abcdef12 | 2/3 (66.7%) | 300 / 50 | 12.0s | v1 baseline |")

        head, sep = run.HISTORY_HEADER.splitlines()
        self.assertEqual(head.count("|"), row.count("|"))
        self.assertEqual(sep.count("|"), row.count("|"))
        self.assertTrue(run.HISTORY_HEADER.endswith("\n"))

        piped = run.history_row({**result, "note": "a|b\nc"})  # a note must not break the table
        self.assertTrue(piped.endswith("| a\\|b c |"))
        self.assertNotIn("\n", piped)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from bench import run

FIXTURES = Path(__file__).parent / "tests_fixtures"
TASKS = FIXTURES / "tasks"
FAKE = FIXTURES / "fake_xolotl.py"
FAKE_CMD = [sys.executable, str(FAKE)]


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

        result = {"run_id": "20261004-1530-82ad7b2", "started": "2026-10-04T15:30:12", "dirty": False,
                  "policy_hash": "abcdef1234567890", "tasks_hash": "0123456789ab", "note": "v1 baseline", "summary": s}
        row = run.history_row(result)
        self.assertEqual(row, "| 2026-10-04 15:30 | 20261004-1530-82ad7b2 | abcdef12 | 0123456789ab | 2/3 (66.7%) "
                              "| 300 / 50 | 12.0s | v1 baseline |")

        head, sep = run.HISTORY_HEADER.splitlines()
        self.assertEqual(head.count("|"), row.count("|"))
        self.assertEqual(sep.count("|"), row.count("|"))
        self.assertTrue(run.HISTORY_HEADER.endswith("\n"))
        self.assertTrue(head.startswith("| Date | Run | Policy | Tasks | Pass |"))

        piped = run.history_row({**result, "note": "a|b\nc"})  # a note must not break the table
        self.assertTrue(piped.endswith("| a\\|b c |"))
        self.assertNotIn("\n", piped)

        dirty = run.history_row({**result, "dirty": True})  # a run from a modified tree says so
        self.assertIn("| 20261004-1530-82ad7b2 (dirty) |", dirty)
        self.assertEqual(dirty.count("|"), row.count("|"))

    def test_summarize_excludes_infra_errors(self):
        def att(task, status):
            return {"task": task, "attempt": 1, "status": status, "pass": status == "pass", "wall_s": 10.0,
                    "tokens": {"prompt": 5, "completion": 1, "cached": 0}}

        attempts = [att("t1", "pass"), att("t1", "pass"), att("t1", "infra_error"), att("t2", "fail"),
                    att("t3", "infra_error"), {"task": "t4", "attempt": 1, "status": "skipped", "pass": False}]
        s = run.summarize(attempts)
        self.assertEqual((s["pass"], s["total"], s["skipped"], s["infra_errors"]), (2, 6, 1, 2))
        self.assertEqual(s["pass_rate"], 0.667)  # 2 of the 3 attempts that really ran
        self.assertEqual(s["per_task"], {"t1": "2/2 (+1 infra)", "t2": "0/1", "t3": "0/0 (+1 infra)", "t4": "skipped"})
        self.assertEqual(s["tokens"]["prompt"], 25)  # an infra attempt may still have cost tokens
        self.assertEqual((s["median_wall_s"], s["total_wall_s"]), (10.0, 30.0))  # but its wall time says nothing

        row = run.history_row({"run_id": "r", "started": "2026-10-04T15:30:12", "tasks_hash": "x", "summary": s})
        self.assertIn("| 2/3 (66.7%) ⚠2 infra |", row)  # the denominator excludes skipped and infra attempts
        clean = run.summarize([att("t1", "pass")])
        self.assertEqual(clean["infra_errors"], 0)
        self.assertNotIn("infra", run.history_row({"run_id": "r", "started": "2026-10-04T15:30:12", "summary": clean}))
        self.assertEqual(run.summarize([att("t", "infra_error")])["pass_rate"], 0.0)  # no division by zero


class Fingerprint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bench-fp-"))
        self.addCleanup(lambda: run._rm(self.tmp))
        self.dir = self.tmp / "fx-ok"
        shutil.copytree(TASKS / "fx-ok", self.dir)
        self.base = run.task_hash(self.dir)

    def copy(self, name: str) -> Path:
        dst = self.tmp / name / "fx-ok"
        shutil.copytree(self.dir, dst)
        return dst

    def test_task_hash_is_short_stable_and_location_independent(self):
        self.assertRegex(self.base, r"^[0-9a-f]{12}$")
        self.assertEqual(run.task_hash(self.copy("elsewhere")), self.base)

    def test_task_hash_ignores_line_endings(self):
        for newline in (b"\n", b"\r\n"):
            copy = self.copy(f"eol{len(newline)}")
            for f in copy.rglob("*"):
                if f.is_file():
                    f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline))
            self.assertEqual(run.task_hash(copy), self.base)  # autocrlf can't change the fingerprint

    def test_task_hash_changes_with_what_the_agent_or_the_checks_see(self):
        for rel in ("verify/check.py", "repo/README.txt", "task.json"):
            with self.subTest(changed=rel):
                copy = self.copy("edit-" + rel.replace("/", "-"))
                with open(copy / rel, "a", encoding="utf-8") as f:
                    f.write("\n# changed\n")
                self.assertNotEqual(run.task_hash(copy), self.base)
        added = self.copy("added")
        (added / "verify" / "extra.py").write_text("x = 1\n")
        self.assertNotEqual(run.task_hash(added), self.base)
        renamed = self.copy("renamed")  # same bytes under another path
        (renamed / "repo" / "README.txt").rename(renamed / "repo" / "README2.txt")
        self.assertNotEqual(run.task_hash(renamed), self.base)

    def test_task_hash_ignores_solution_and_pycache(self):
        (self.dir / "solution" / "answer.txt").write_text("43")
        (self.dir / "solution" / "new.py").write_text("x = 1\n")
        (self.dir / "verify" / "__pycache__").mkdir()
        (self.dir / "verify" / "__pycache__" / "check.cpython-314.pyc").write_bytes(b"\x00\x01")
        self.assertEqual(run.task_hash(self.dir), self.base)

    def test_suite_hash_is_order_independent_and_sensitive(self):
        h = run.suite_hash({"a": "1" * 12, "b": "2" * 12})
        self.assertRegex(h, r"^[0-9a-f]{12}$")
        self.assertEqual(run.suite_hash({"b": "2" * 12, "a": "1" * 12}), h)
        self.assertNotEqual(run.suite_hash({"a": "1" * 12, "b": "3" * 12}), h)
        self.assertNotEqual(run.suite_hash({"a": "1" * 12}), h)
        self.assertNotEqual(run.suite_hash({"a": "2" * 12, "b": "1" * 12}), h)  # the id is part of the line


def pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process there, so ask the OS instead
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FrozenClock(datetime):
    """Stands in for `run.datetime`: every run starts at the same minute."""
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 4, 15, 30, 12).astimezone()


def fx_task(**over) -> dict:
    """The fx-ok fixture task as loaded from disk, with some fields overridden."""
    return {**run.load_task(TASKS / "fx-ok"), **over}


class Orchestration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bench-test-"))
        self.addCleanup(lambda: run._rm(self.tmp))  # lazy: git leaves read-only files that plain rmtree can't delete
        self.home = self.tmp / "home"
        self.home.mkdir()

    def attempt(self, prompt: str, keep: bool = False, runner_budget: int | None = None, **task_over) -> dict:
        res = run.run_attempt(fx_task(prompt=prompt, **task_over), TASKS / "fx-ok", FAKE_CMD, self.home, runner_budget, keep)
        if res["kept_dir"]:
            self.addCleanup(lambda p=Path(res["kept_dir"]): run._rm(p))
        return res

    @contextlib.contextmanager
    def track_mkdtemp(self):
        """Yield the list of temp dirs the runner creates while the block runs."""
        made, real = [], tempfile.mkdtemp

        def spy(*a, **k):
            made.append(real(*a, **k))
            return made[-1]

        with mock.patch.object(run.tempfile, "mkdtemp", spy):
            yield made

    def quiet_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run.main(["--home", str(self.home), *argv])  # a temp home: never read the real ~/.xolotl
        return code, out.getvalue()

    def make_tasks(self, prompts: dict[str, str]) -> Path:
        """A tasks dir holding a copy of fx-ok per entry, each with its own id and fake-xolotl prompt."""
        root = self.tmp / "tasks"
        root.mkdir()
        for name, prompt in prompts.items():
            shutil.copytree(TASKS / "fx-ok", root / name)
            task_json = root / name / "task.json"
            task_json.write_text(json.dumps({**json.loads(task_json.read_text()), "id": name, "prompt": prompt}))
        return root

    def test_pass_path(self):
        made = []
        real_mkdtemp = tempfile.mkdtemp

        def spy(*a, **k):
            made.append(real_mkdtemp(*a, **k))
            return made[-1]

        with mock.patch.object(run.tempfile, "mkdtemp", spy):
            res = self.attempt("ok")
        self.assertEqual((res["task"], res["attempt"], res["status"], res["pass"]), ("fx-ok", 1, "pass", True))
        self.assertEqual((res["exit_code"], res["timed_out"]), (0, False))
        self.assertEqual(res["model_calls"], 1)
        self.assertEqual(res["tokens"], {"prompt": 10, "completion": 5, "cached": 2})
        self.assertEqual((res["policy_hash"], res["model"]), ("fakehash", "fake-model"))
        self.assertIsInstance(res["wall_s"], float)
        self.assertIsNone(res["kept_dir"])  # a passing attempt's temp dir is deleted ...
        self.assertEqual(len(made), 1)
        self.assertFalse(os.path.exists(made[0]))  # ... including git's read-only objects
        self.assertIsNone(res["session_dir"])  # the fake writes no session
        self.assertIn("fake: ok", res["stderr_tail"])  # Xolotl's stderr is kept, not just drained
        self.assertEqual(res["task_hash"], run.task_hash(TASKS / "fx-ok"))
        self.assertEqual(res["infra_reason"], "")

    def test_keep_flag_and_git_setup(self):
        res = self.attempt("ok", keep=True)
        self.assertEqual(res["status"], "pass")
        kept = Path(res["kept_dir"])
        self.assertFalse((kept / "verify").exists())  # the hidden checks never stay behind in a kept dir
        self.assertTrue((kept / "answer.txt").is_file())
        self.assertTrue((kept / "README.txt").is_file())
        log = subprocess.run(["git", "log", "--format=%an <%ae> %s"], cwd=kept, capture_output=True, text=True, check=True)
        self.assertEqual(log.stdout.strip(), "bench <bench@local> init")  # one commit, made by the runner

    def test_planted_verify_is_replaced(self):
        res = self.attempt("plant")
        self.assertEqual((res["status"], res["pass"]), ("fail", False))  # the planted always-passing file is not what ran
        kept = Path(res["kept_dir"])
        self.assertTrue(kept.is_dir())  # a failed attempt's dir is kept for inspection
        self.assertFalse((kept / "verify").exists())  # neither the planted file nor the real checks are left in a kept dir
        self.assertIn("answer.txt", res["verify_output"])  # the real check ran (not the planted one) and failed on the missing file

    def test_timeout_kills_tree_and_returns_promptly(self):
        t0 = time.monotonic()
        res = self.attempt("hang", timeout_s=3)
        elapsed = time.monotonic() - t0
        self.assertTrue(res["timed_out"])
        self.assertEqual((res["status"], res["pass"]), ("fail", False))
        self.assertLess(elapsed, 15)
        self.assertGreaterEqual(res["wall_s"], 3)
        pid = int(Path(res["kept_dir"], "child.pid").read_text())
        self.addCleanup(lambda: pid_alive(pid) and os.kill(pid, 9))  # never leak the 60 s sleeper if the assert below fails
        for _ in range(30):
            if not pid_alive(pid):
                break
            time.sleep(0.1)
        self.assertFalse(pid_alive(pid), "the grandchild survived the tree kill")

    def test_garbage_stdout(self):
        res = self.attempt("garbage")  # 1 MB of non-UTF-8, non-JSON bytes must neither block the child nor raise
        self.assertEqual(res["model_calls"], 1)
        self.assertEqual(res["status"], "fail")
        self.assertFalse(res["timed_out"])
        self.assertLessEqual(len(res["stderr_tail"].encode()), 1024)  # 5 KB on stderr, last 1 KB kept
        self.assertTrue(res["stderr_tail"].rstrip().endswith("END"))

    def test_no_session_start_is_infra_error(self):
        res = self.attempt("crash")  # exits 1 without a single event, like Xolotl with no API key
        self.assertEqual((res["status"], res["pass"], res["exit_code"]), ("infra_error", False, 1))
        self.assertIn("session_start", res["infra_reason"])
        self.assertIn("ZAI_API_KEY", res["stderr_tail"])
        self.assertTrue(Path(res["kept_dir"]).is_dir())  # kept for inspection, like a failure

    def test_llm_error_is_infra_error(self):
        res = self.attempt("llmerror")  # quota exhausted: nothing the agent could have done
        self.assertEqual((res["status"], res["pass"], res["errors"]), ("infra_error", False, 1))
        self.assertIn("quota", res["infra_reason"])

    def test_agent_side_llm_errors_are_not_infra_errors(self):
        res = self.attempt("llmretry")  # a context overflow the agent compacted away and retried
        self.assertEqual((res["status"], res["infra_reason"]), ("pass", ""))
        res = self.attempt("llmoverflow")  # still too long after compaction: the agent's own failure
        self.assertEqual((res["status"], res["infra_reason"]), ("fail", ""))
        res = self.attempt("llmlate")  # a measured pass stays a pass even if the API failed afterwards
        self.assertEqual((res["status"], res["errors"], res["infra_reason"]), ("pass", 1, ""))

    def test_missing_toolchain_skips(self):
        boom = AssertionError("must not start any subprocess or temp dir")
        with mock.patch.object(run.subprocess, "Popen", side_effect=boom), \
                mock.patch.object(run.subprocess, "run", side_effect=boom), \
                mock.patch.object(run.tempfile, "mkdtemp", side_effect=boom):
            res = self.attempt("ok", requires=["definitely-not-a-real-exe"])
        self.assertEqual((res["status"], res["pass"], res["kept_dir"]), ("skipped", False, None))
        self.assertIn("definitely-not-a-real-exe", res["verify_output"])
        self.assertEqual(res["tokens"], {"prompt": 0, "completion": 0, "cached": 0})  # summarize() can read it
        self.assertEqual(res["stderr_tail"], "")

    def test_budget_flag_and_task_override(self):
        def xolotl_argv(runner_budget, **task_over):
            with mock.patch.object(run.subprocess, "Popen", wraps=subprocess.Popen) as popen:
                self.attempt("ok", runner_budget=runner_budget, **task_over)
            (argv,) = [c.args[0] for c in popen.call_args_list if isinstance(c.args[0], list) and "-p" in c.args[0]]
            return argv

        argv = xolotl_argv(None)
        self.assertNotIn("--budget", argv)
        self.assertEqual(argv[argv.index("--mode") + 1], "yolo")
        self.assertEqual(xolotl_argv(123)[-2:], ["--budget", "123"])
        self.assertEqual(xolotl_argv(123, budget=500)[-2:], ["--budget", "500"])  # a task's own budget beats the runner's

    def test_selfcheck(self):
        ok, reason = run.selfcheck(run.load_task(TASKS / "fx-ok"), TASKS / "fx-ok")
        self.assertTrue(ok, reason)

        def variant(name: str, edit) -> tuple[bool, str]:
            root = self.tmp / name
            shutil.copytree(TASKS / "fx-ok", root / "fx-ok")
            edit(root / "fx-ok")
            return run.selfcheck(run.load_task(root / "fx-ok"), root / "fx-ok")

        ok, reason = variant("badsol", lambda d: (d / "solution" / "answer.txt").write_text("41\n"))
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("solution fails"), reason)

        ok, reason = variant("solved", lambda d: (d / "repo" / "answer.txt").write_text("42\n"))
        self.assertEqual((ok, reason), (False, "unsolved repo already passes"))

        ok, reason = variant("nosol", lambda d: shutil.rmtree(d / "solution"))
        self.assertFalse(ok)
        self.assertIn("solution", reason)

        # a task whose toolchain is missing is skipped, not failed
        ok, reason = run.selfcheck(fx_task(requires=["definitely-not-a-real-exe"]), TASKS / "fx-ok")
        self.assertTrue(ok)
        self.assertIn("skipped", reason)

    def test_find_bash(self):
        bash = run.find_bash()
        self.assertNotIn("system32", " ".join(bash).lower())  # never the WSL stub
        if os.name == "nt":
            self.assertEqual(Path(bash[0]).name.lower(), "bash.exe")
            self.assertTrue(Path(bash[0]).is_file())

    def test_main_writes_results_and_history(self):
        results = self.tmp / "results"
        results.mkdir()
        history = results / "HISTORY.md"
        history.write_text(run.HISTORY_HEADER + "| 2026-01-01 00:00 | old | abc | def | 1/1 (100.0%) | 1 / 1 | 1.0s | old |\n")
        before = history.read_text().splitlines()
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": "sentinel-secret-123"}):
            code, out = self.quiet_main("--tasks-dir", str(TASKS), "--xolotl", str(FAKE), "--note", "t",
                                        "--results-dir", str(results))
        self.assertEqual(code, 0)
        self.assertNotIn("sentinel-secret-123", out)  # the runner never prints environment values

        (path,) = results.glob("*.json")
        self.assertNotIn("sentinel-secret-123", path.read_text())
        res = json.loads(path.read_text())
        self.assertRegex(res["run_id"], r"^\d{8}-\d{4}-([0-9a-f]{7}|unknown)$")
        self.assertEqual(path.stem, res["run_id"])
        self.assertEqual(res["summary"]["pass"], 1)
        self.assertEqual((res["summary"]["total"], res["summary"]["per_task"]), (1, {"fx-ok": "1/1"}))
        self.assertEqual((res["xolotl_version"], res["model"], res["policy_hash"]), ("xolotl-fake 0.0.0", "fake-model", "fakehash"))
        self.assertEqual((res["repeat"], res["budget"], res["note"], res["partial"]), (1, None, "t", False))
        self.assertIsInstance(res["dirty"], bool)
        self.assertEqual(len(res["git_sha"]), 40)
        self.assertIsNotNone(datetime.fromisoformat(res["started"]).tzinfo)  # local time with offset
        (att,) = res["attempts"]
        self.assertEqual((att["task"], att["attempt"], att["status"], att["model_calls"]), ("fx-ok", 1, "pass", 1))
        self.assertRegex(out, r"fx-ok.*pass")  # one progress line per attempt

        fx_hash = run.task_hash(TASKS / "fx-ok")  # the suite fingerprint, per attempt and per run
        self.assertEqual(att["task_hash"], fx_hash)
        self.assertEqual(res["tasks_hash"], run.suite_hash({"fx-ok": fx_hash}))
        self.assertEqual((res["xolotl_source"], res["xolotl_config"]), (str(FAKE), None))  # no config.json in the temp home
        self.assertEqual(res["summary"]["infra_errors"], 0)

        after = history.read_text().splitlines()
        self.assertEqual(after[:len(before)], before)
        self.assertEqual(len(after), len(before) + 1)  # exactly one new row
        self.assertTrue(after[-1].endswith("| t |"), after[-1])
        self.assertIn(res["run_id"], after[-1])
        self.assertIn(f"| {res['tasks_hash']} |", after[-1])
        self.assertEqual(" (dirty) |" in after[-1], res["dirty"])  # the row marks a run from a modified tree
        self.assertEqual(after[-1].count("|"), run.HISTORY_HEADER.splitlines()[0].count("|"))

        # a missing HISTORY.md gets the header first
        fresh = self.tmp / "fresh"
        code, _ = self.quiet_main("--tasks-dir", str(TASKS), "--xolotl", str(FAKE), "--repeat", "2", "--results-dir", str(fresh))
        self.assertEqual(code, 0)
        rows = (fresh / "HISTORY.md").read_text().splitlines()
        self.assertEqual("\n".join(rows[:2]) + "\n", run.HISTORY_HEADER)
        self.assertEqual(len(rows), 3)
        (path,) = fresh.glob("*.json")
        self.assertEqual(json.loads(path.read_text())["summary"]["total"], 2)

    def test_main_records_xolotl_config(self):
        (self.home / ".xolotl").mkdir()
        config = self.home / ".xolotl" / "config.json"
        config.write_bytes(b'\xef\xbb\xbf{"model": "m1", "mode": "yolo", "api_key_env": "BENCH_TEST_KEY"}')  # BOM, as PowerShell writes
        argv = ["--tasks-dir", str(TASKS), "--xolotl", str(FAKE)]
        with mock.patch.dict(os.environ, {"BENCH_TEST_KEY": "sentinel-secret-456"}):  # only the variable's name is in the config
            code, out = self.quiet_main(*argv, "--results-dir", str(self.tmp / "r1"))
        self.assertEqual(code, 0)
        (path,) = (self.tmp / "r1").glob("*.json")
        self.assertEqual(json.loads(path.read_text())["xolotl_config"],
                         {"model": "m1", "mode": "yolo", "api_key_env": "BENCH_TEST_KEY"})
        self.assertNotIn("sentinel-secret-456", path.read_text() + out)  # environment values are never read

        config.write_text("{not json")  # a broken config is recorded as absent, not a crash
        self.assertEqual(self.quiet_main(*argv, "--results-dir", str(self.tmp / "r2"))[0], 0)
        (path,) = (self.tmp / "r2").glob("*.json")
        self.assertIsNone(json.loads(path.read_text())["xolotl_config"])

    def test_main_infra_errors_are_excluded_and_flagged(self):
        tasks = self.make_tasks({"fx-1": "ok", "fx-2": "crash"})
        results = self.tmp / "results"
        code, out = self.quiet_main("--tasks-dir", str(tasks), "--xolotl", str(FAKE), "--results-dir", str(results))
        self.assertEqual(code, 0)  # the run completed; the warning is in the output and the row
        (path,) = results.glob("*.json")
        res = json.loads(path.read_text())
        self.assertEqual([a["status"] for a in res["attempts"]], ["pass", "infra_error"])
        s = res["summary"]
        self.assertEqual((s["pass"], s["total"], s["infra_errors"], s["pass_rate"]), (1, 2, 1, 1.0))
        self.assertEqual(s["per_task"], {"fx-1": "1/1", "fx-2": "0/0 (+1 infra)"})
        self.assertRegex(out, r"fx-2 #1: infra_error")
        self.assertRegex(out, r"(?i)warning.*1.*infra")
        row = (results / "HISTORY.md").read_text(encoding="utf-8").splitlines()[-1]  # the file is UTF-8
        self.assertIn("| 1/1 (100.0%) ⚠1 infra |", row)

    def test_main_fingerprint(self):
        boom = AssertionError("--fingerprint must not build, run Xolotl or write results")
        results = self.tmp / "results"
        with mock.patch.object(run.subprocess, "Popen", side_effect=boom), \
                mock.patch.object(run.subprocess, "run", side_effect=boom):
            code, out = self.quiet_main("--fingerprint", "--tasks-dir", str(TASKS), "--results-dir", str(results))
        self.assertEqual(code, 0)
        h = run.task_hash(TASKS / "fx-ok")
        self.assertRegex(out, rf"fx-ok\s+{h}")
        self.assertRegex(out, rf"tasks_hash\s+{run.suite_hash({'fx-ok': h})}")
        self.assertFalse(results.exists())

    def test_main_ctrl_c_writes_partial_results(self):
        results = self.tmp / "results"
        real = run.run_attempt
        calls = []

        def interrupt_second(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(*a, **k)

        with mock.patch.object(run, "run_attempt", interrupt_second):
            code, _ = self.quiet_main("--tasks-dir", str(TASKS), "--xolotl", str(FAKE), "--repeat", "3",
                                      "--results-dir", str(results))
        self.assertEqual(code, 130)
        (path,) = results.glob("*.json")
        res = json.loads(path.read_text())
        self.assertTrue(res["partial"])
        self.assertEqual(len(res["attempts"]), 1)  # only the attempt that finished
        self.assertFalse((results / "HISTORY.md").exists())  # a partial run never reaches HISTORY.md

    def test_run_id_collision_gets_a_suffix(self):
        results = self.tmp / "results"
        fixed_git = lambda *a: "a" * 40 if a[0] == "rev-parse" else ""  # noqa: E731
        with mock.patch.object(run, "datetime", FrozenClock), mock.patch.object(run, "_git", fixed_git):
            for note in ("first", "second", "third"):  # same minute, same sha
                code, _ = self.quiet_main("--tasks-dir", str(TASKS), "--xolotl", str(FAKE), "--note", note,
                                          "--results-dir", str(results))
                self.assertEqual(code, 0)
        base = "20261004-1530-aaaaaaa"
        self.assertEqual({p.name for p in results.glob("*.json")}, {f"{base}.json", f"{base}-2.json", f"{base}-3.json"})
        for suffix, note in (("", "first"), ("-2", "second"), ("-3", "third")):  # nothing was overwritten
            res = json.loads((results / f"{base}{suffix}.json").read_text())
            self.assertEqual((res["run_id"], res["note"]), (base + suffix, note))
        rows = (results / "HISTORY.md").read_text().splitlines()[2:]
        self.assertEqual([r.split("|")[2].strip() for r in rows], [base, f"{base}-2", f"{base}-3"])

    def test_exception_in_an_attempt_keeps_earlier_attempts(self):
        tasks = self.make_tasks({"fx-1": "ok", "fx-2": "ok"})
        shutil.rmtree(tasks / "fx-2" / "repo")  # the second task's repo copy raises FileNotFoundError
        results = self.tmp / "results"
        err = io.StringIO()
        with self.track_mkdtemp() as made, contextlib.redirect_stderr(err):
            code, _ = self.quiet_main("--tasks-dir", str(tasks), "--xolotl", str(FAKE), "--results-dir", str(results))
        self.assertEqual(code, 1)
        (path,) = results.glob("*.json")
        res = json.loads(path.read_text())
        self.assertTrue(res["partial"])
        self.assertEqual([a["task"] for a in res["attempts"]], ["fx-1"])  # the finished attempt survives
        self.assertEqual(res["summary"]["pass"], 1)
        self.assertIn("fx-2", res["error"])
        self.assertIn("FileNotFoundError", res["error"])
        self.assertIn("Traceback", err.getvalue())
        self.assertFalse((results / "HISTORY.md").exists())  # a partial run never reaches HISTORY.md
        self.assertEqual(len(made), 2)
        self.assertFalse(any(os.path.exists(p) for p in made))  # neither attempt left a temp dir behind

    def test_cleanup_error_does_not_mask_interrupt(self):
        with self.track_mkdtemp() as made, \
                mock.patch.object(run, "_run_tree", side_effect=KeyboardInterrupt), \
                mock.patch.object(run, "_rm", side_effect=OSError("locked")):  # e.g. a dying child still holds the dir
            with self.assertRaises(KeyboardInterrupt):
                run.run_attempt(fx_task(prompt="ok"), TASKS / "fx-ok", FAKE_CMD, self.home, None, False)
        self.addCleanup(lambda: run._rm(Path(made[0])))  # the patched _rm left the dir behind

    def test_dirty_ignores_results_dir(self):
        repo = self.tmp / "repo"
        (repo / "bench" / "results").mkdir(parents=True)
        (repo / "bench" / "results" / "HISTORY.md").write_text("header\n")
        (repo / "tracked.txt").write_text("x")
        run._init_repo(str(repo))
        with mock.patch.object(run, "REPO_ROOT", repo):
            self.assertFalse(run._dirty())
            (repo / "bench" / "results" / "HISTORY.md").write_text("header\nrow\n")  # tracked file modified by a run
            (repo / "bench" / "results" / "20261004-1530-abc.json").write_text("{}")  # untracked results file
            self.assertFalse(run._dirty())
            (repo / "other.txt").write_text("y")  # any other change counts
            self.assertTrue(run._dirty())
            (repo / "other.txt").unlink()
            self.assertFalse(run._dirty())
            (repo / "tracked.txt").write_text("changed")
            self.assertTrue(run._dirty())

    def test_main_selfcheck_and_errors(self):
        code, out = self.quiet_main("--selfcheck", "--tasks-dir", str(TASKS))
        self.assertEqual(code, 0)
        self.assertRegex(out, r"ok\s+fx-ok")

        root = self.tmp / "tasks"
        shutil.copytree(TASKS, root)
        (root / "fx-ok" / "repo" / "answer.txt").write_text("42")  # already solved
        code, out = self.quiet_main("--selfcheck", "--tasks-dir", str(root))
        self.assertEqual(code, 1)
        self.assertRegex(out, r"FAIL\s+fx-ok")

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(run.main(["--tasks-dir", str(TASKS), "--tasks", "nope", "--xolotl", str(FAKE)]), 2)
        self.assertIn("nope", err.getvalue())


if __name__ == "__main__":
    unittest.main()

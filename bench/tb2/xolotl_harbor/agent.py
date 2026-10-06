"""Harbor installed agent that runs Xolotl (headless, --mode yolo) inside a Terminal-Bench task container.

Importable as `xolotl_harbor.agent:XolotlAgent` once bench/tb2 is on Harbor's PYTHONPATH (see README). The static musl
binary is built by build_linux.py and uploaded in install(); tokens come from the JSON-lines stream Xolotl writes to
/logs/agent, which Harbor maps to <trial>/agent/ on the host.
"""
import json
import os
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import core


class XolotlAgent(BaseInstalledAgent):
    def __init__(self, *args, binary: str | None = None, agent_timeout_sec: float | None = None, **kwargs):
        # `binary` and `agent_timeout_sec` arrive as `--ak binary=... --ak agent_timeout_sec=...`; never the API key.
        super().__init__(*args, **kwargs)
        self._binary = binary
        self._agent_timeout_sec = agent_timeout_sec  # optional override; normally run() reads the task's own timeout
        if agent_timeout_sec is not None:
            core.resolve_timeout(agent_timeout_sec, None)  # a typo'd override fails now, not after the container is up

    @staticmethod
    def name() -> str:
        return "xolotl"

    def get_version_command(self) -> str:
        return f"{core.BINARY} --version"

    async def install(self, environment: BaseEnvironment) -> None:
        path = self._binary or os.environ.get("XOLOTL_LINUX_BIN")
        if not path or not Path(path).is_file():
            raise RuntimeError(f"xolotl linux binary not found ({path!r}): build it with bench/tb2/build_linux.py and "
                               "pass --ak binary=<path> or set XOLOTL_LINUX_BIN")
        await environment.upload_file(Path(path), core.BINARY)
        await self.exec_as_root(environment, f"chmod 755 {core.BINARY}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        key = core.read_api_key()
        if not key:
            raise RuntimeError(f"{core.KEY_ENV} is not set (process env or Windows User scope)")
        # The task's own agent timeout (read from its task.toml; Harbor doesn't pass it) minus 60 s, so Xolotl stops first.
        rec = core.resolve_timeout(self._agent_timeout_sec, environment.environment_dir)
        try:  # which timeout this trial really got; run.py puts it in the results. Losing it must not lose the trial.
            (self.logs_dir / core.TIMEOUT_FILE).write_text(json.dumps(rec) + "\n", encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"could not record {core.TIMEOUT_FILE}: {e}")
        cmd = core.run_command(instruction, core.model_arg(self.model_name), rec["budget_s"])
        # The key goes only through env=; the command always exits 0, so the verifier decides pass or fail.
        await self.exec_as_agent(environment, cmd, env={core.KEY_ENV: key})

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Runs after Harbor copied /logs/agent to the host, and only while the context is still empty."""
        try:
            text = (self.logs_dir / "xolotl.jsonl").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        m = core.metrics_from_jsonl(text)
        try:
            exit_code = int((self.logs_dir / "exit_code").read_text().strip())
        except (OSError, ValueError):
            exit_code = None
        tokens = m["tokens"]
        context.n_input_tokens = tokens["prompt"]
        context.n_cache_tokens = tokens["cached"]
        context.n_output_tokens = tokens["completion"]
        context.metadata = {
            "model_calls": m["model_calls"], "tool_calls": m["tool_calls"], "compactions": m["compactions"],
            "policy_hash": m["policy_hash"], "exit_code": exit_code, "infra_reason": m["infra_reason"],
        }

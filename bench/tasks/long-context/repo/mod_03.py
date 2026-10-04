"""Start-up defaults."""

from config import CONFIG

__all__ = [
    "load_defaults",
    "describe_env",
    "parse_pair",
    "retry_budget",
    "needs_defaults",
]


def load_defaults():
    """Register the built-in defaults and report how many settings exist."""
    CONFIG.update({"env": "dev", "debug": False, "retries": 3, "timeout": 30})
    return len(CONFIG)


def describe_env():
    """One-line description of the current environment."""
    return f"{CONFIG.get('env', 'dev')} (debug={CONFIG.get('debug', False)})"


def parse_pair(text):
    """Split 'key=value' into a stripped (key, value) pair."""
    key, _, value = text.partition("=")
    return key.strip(), value.strip()


def retry_budget(spent):
    """How many retries are left after `spent` attempts."""
    return max(0, CONFIG.get("retries", 3) - spent)


def needs_defaults():
    """True while nothing has been registered yet."""
    return not CONFIG

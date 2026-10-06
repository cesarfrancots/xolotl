"""Runtime switches."""

from config import CONFIG

__all__ = [
    "set_log_level",
    "bump_generation",
    "defaults_for_env",
    "log_enabled",
    "current_generation",
]


def set_log_level(level):
    """Change the active log level."""
    CONFIG["log_level"] = level.upper()


def bump_generation():
    """Advance the settings generation counter and return it."""
    CONFIG["generation"] = CONFIG.get("generation", 0) + 1
    return CONFIG["generation"]


def defaults_for_env(env):
    """Build the starting settings for one environment."""
    config = {"env": env, "debug": env == "dev"}
    config["log_level"] = "DEBUG" if config["debug"] else "INFO"
    config.setdefault("retries", 3)
    return config


def log_enabled(level):
    """True if `level` is at least as severe as the configured one."""
    order = ["DEBUG", "INFO", "WARNING", "ERROR"]
    return order.index(level) >= order.index(CONFIG.get("log_level", "INFO"))


def current_generation():
    """The settings generation counter, 0 before the first change."""
    return CONFIG.get("generation", 0)

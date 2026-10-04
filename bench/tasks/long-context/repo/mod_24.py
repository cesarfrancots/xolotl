"""Lifecycle."""

from config import CONFIG

__all__ = [
    "reset_all",
    "is_configured",
    "setting_count",
    "shutdown_order",
    "settings_summary",
]


def reset_all():
    """Forget every setting (used between test runs)."""
    CONFIG.clear()


def is_configured():
    """True once any setting has been registered."""
    return bool(CONFIG)


def setting_count():
    """Number of registered settings."""
    return len(CONFIG)


def shutdown_order(names):
    """Stop things in the reverse of the order they started."""
    return sorted(names, reverse=True)


def settings_summary():
    """All settings as a sorted 'key=value, ...' line."""
    return ", ".join(f"{key}={value}" for key, value in sorted(CONFIG.items()))

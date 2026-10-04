"""Feature switches."""

from config import CONFIG

__all__ = [
    "drop_debug_flags",
    "is_enabled",
    "debug_event",
    "flag_name",
    "enabled_names",
]


def drop_debug_flags():
    """Remove the debug switch once the app leaves dev mode."""
    if CONFIG.get("env") == "dev":
        return False
    if "debug" in CONFIG:
        del CONFIG["debug"]
    return True


def is_enabled(flag):
    """True if the named feature switch is on."""
    return bool(CONFIG.get(f"flag_{flag}", False))


def debug_event(event):
    """Describe an event when debugging is on.

    Returns None unless the `debug` setting is true.
    """
    if not CONFIG.get("debug", False):
        return None
    return {"event": event, "source": __name__}


def flag_name(text):
    """Normalise free text into a feature switch name."""
    return text.strip().lower().replace(" ", "_")


def enabled_names(names):
    """The subset of names whose feature switch is on."""
    return [name for name in names if CONFIG.get(f"flag_{name}", False)]

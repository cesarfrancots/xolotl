"""Timeouts."""

from config import CONFIG as SETTINGS

__all__ = [
    "ensure_timeout",
    "remaining",
    "to_millis",
    "timed_out",
    "deadline",
]


def ensure_timeout(seconds=30):
    """Make sure a timeout is configured and return the effective one."""
    return SETTINGS.setdefault("timeout", seconds)


def remaining(elapsed):
    """Seconds left before the configured timeout."""
    return max(0, SETTINGS.get("timeout", 30) - elapsed)


def to_millis(seconds):
    """Convert seconds to whole milliseconds."""
    return int(seconds * 1000)


def timed_out(elapsed):
    """True once `elapsed` exceeds the configured timeout."""
    return elapsed > SETTINGS.get("timeout", 30)


def deadline(start):
    """The moment at which the configured timeout runs out."""
    return start + SETTINGS.get("timeout", 30)

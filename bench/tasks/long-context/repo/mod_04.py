"""Retry policy."""

from config import CONFIG

__all__ = [
    "attempts_left",
    "backoff_delay",
    "is_retryable",
    "group_failures",
    "jittered",
]


def attempts_left(done):
    """Attempts remaining out of the configured maximum."""
    return max(0, CONFIG.get("max_attempts", 5) - done)


def backoff_delay(attempt):
    """Exponential backoff delay in seconds for the given attempt number."""
    base = CONFIG.get("backoff_base", 0.5)
    return base * 2 ** attempt


def is_retryable(status):
    """True for HTTP statuses worth retrying."""
    return status in (429, 500, 502, 503, 504)


def group_failures(pairs):
    """Collect the failure messages for each target."""
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups


def jittered(delay, fraction):
    """Stretch a delay by half of `fraction` to spread out retries."""
    return delay * (1 + fraction / 2)

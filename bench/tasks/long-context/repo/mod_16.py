"""Event bus."""

from config import CONFIG

__all__ = [
    "fan_out",
    "by_topic",
    "topic_of",
    "capped_queue",
    "matches",
]


def fan_out(items):
    """Split handlers into groups that run together.

    The group size comes from the `fanout` setting (default 4).
    """
    size = CONFIG.get("fanout", 4)
    return [items[i:i + size] for i in range(0, len(items), size)]


def by_topic(pairs):
    """Collect event payloads by topic."""
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups


def topic_of(event_name):
    """The topic part of a dotted event name."""
    return event_name.split(".", 1)[0]


def capped_queue(value):
    """Clamp a queue length to the configured maximum.

    The result is the smaller of `value` and the `max_events` setting, which defaults to 1000.
    """
    limit = CONFIG.get("max_events", 1000)
    return min(value, limit)


def matches(pattern, topic):
    """True if a topic matches a pattern such as 'orders.*'."""
    return pattern == "*" or topic.startswith(pattern.rstrip("*"))

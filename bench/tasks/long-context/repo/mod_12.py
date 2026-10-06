"""Task scheduler."""

from config import CONFIG

__all__ = [
    "scaled_delays",
    "with_overrides",
    "by_hour",
    "next_slot",
    "first_runnable",
]


def scaled_delays(values):
    """Multiply delays by the configured schedule scale.

    The factor is the `schedule_scale` setting, or 1 when it is not set.
    """
    factor = CONFIG.get("schedule_scale", 1)
    return [v * factor for v in values]


def with_overrides(extra):
    """Settings plus the given overrides."""
    merged = dict(CONFIG)
    merged.update(extra)
    return merged


def by_hour(pairs):
    """Collect task names by the hour they run."""
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups


def next_slot(now, step):
    """The next multiple of `step` after `now`."""
    return now + step - (now % step)


def first_runnable(tasks):
    """The tasks that may run now, limited by the configured parallelism."""
    limit = CONFIG.get("max_parallel", 4)
    return tasks[:limit]

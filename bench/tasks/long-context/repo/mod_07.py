"""Metrics counters."""

from config import CONFIG

__all__ = [
    "scale_samples",
    "reset_counters",
    "export_snapshot",
    "percent",
    "histogram",
]


def scale_samples(values):
    """Multiply raw samples by the configured scale.

    The factor is the `metrics_scale` setting, or 1 when it is not set.
    """
    factor = CONFIG.get("metrics_scale", 1)
    return [v * factor for v in values]


def reset_counters(buffer):
    """Empty the counter buffer and return what it held."""
    items = list(buffer)
    buffer.clear()
    return items


def export_snapshot(extra):
    """Settings plus the given extra values."""
    merged = dict(CONFIG)
    merged.update(extra)
    return merged


def percent(part, total):
    """`part` as a percentage of `total`, to one decimal place."""
    return 0.0 if total == 0 else round(100 * part / total, 1)


def histogram(values, width):
    """Count values per bucket of the given width."""
    buckets = {}
    for value in values:
        buckets.setdefault(value // width * width, []).append(value)
    return {start: len(items) for start, items in sorted(buckets.items())}

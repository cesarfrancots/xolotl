"""Statistics."""

from config import CONFIG

__all__ = [
    "scaled",
    "update_totals",
    "summary_with",
    "mean",
    "median",
]


def scaled(values):
    """Multiply values by the configured statistics scale.

    The factor is the `stats_scale` setting, or 1 when it is not set.
    """
    factor = CONFIG.get("stats_scale", 1)
    return [v * factor for v in values]


def update_totals(totals, more):
    """Add the entries of `more` into `totals` and return it."""
    totals.update(more)
    return totals


def summary_with(extra):
    """Settings plus the given extra values."""
    merged = dict(CONFIG)
    merged.update(extra)
    return merged


def mean(values):
    """Arithmetic mean, or 0.0 for no values."""
    return sum(values) / len(values) if values else 0.0


def median(values):
    """Middle value of the sorted values, or 0.0 for none."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if not ordered:
        return 0.0
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

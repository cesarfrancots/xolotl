"""Work queue helpers."""

from config import CONFIG

__all__ = [
    "batches",
    "drain",
    "is_blank",
    "priority_label",
    "oldest",
]


def batches(items):
    """Split items into batches of the configured size.

    The group size comes from the `batch_size` setting (default 10).
    """
    size = CONFIG.get("batch_size", 10)
    return [items[i:i + size] for i in range(0, len(items), size)]


def drain(buffer):
    """Empty the buffer and return what it held."""
    items = list(buffer)
    buffer.clear()
    return items


def is_blank(text):
    """True for empty or whitespace-only text."""
    return not text or not text.strip()


def priority_label(score):
    """Label a score as high or low priority.

    The cut-off is the `priority_cutoff` setting (default 5).
    """
    threshold = CONFIG.get("priority_cutoff", 5)
    return "high" if score >= threshold else "low"


def oldest(items):
    """The (timestamp, payload) pair with the smallest timestamp, or None."""
    return min(items, key=lambda item: item[0]) if items else None

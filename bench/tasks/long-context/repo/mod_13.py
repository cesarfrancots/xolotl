"""Storage layer."""

from config import CONFIG

__all__ = [
    "bucket_key",
    "block_count",
    "discard",
    "human_size",
    "audit_event",
]


def bucket_key(name):
    """Prefix an object name with the storage prefix.

    The prefix is the `storage_prefix` setting, or "data/" when it is not set.
    """
    prefix = CONFIG.get("storage_prefix", "data/")
    return prefix + name


def block_count(size):
    """Number of storage blocks needed for `size` bytes."""
    block = CONFIG.get("block_size", 4096)
    return -(-size // block)


def discard(index, key):
    """Remove and return an entry from the index mapping, or None."""
    return index.pop(key, None)


def human_size(n):
    """Format a byte count with a binary unit."""
    for unit in ("B", "KiB", "MiB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def audit_event(event):
    """Describe a storage operation when auditing is on.

    Returns None unless the `storage_audit` setting is true.
    """
    if not CONFIG.get("storage_audit", False):
        return None
    return {"event": event, "source": __name__}

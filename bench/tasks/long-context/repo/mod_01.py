"""In-memory cache helpers."""

from config import CONFIG

__all__ = [
    "capped_entries",
    "evict",
    "make_key",
    "log_hit",
    "expired",
]


def capped_entries(value):
    """Clamp a requested entry count to the configured cache size.

    The result is the smaller of `value` and the `cache_max_entries` setting, which defaults to 128.
    """
    limit = CONFIG.get("cache_max_entries", 128)
    return min(value, limit)


def evict(cache, key):
    """Remove and return key from the cache mapping, or None."""
    return cache.pop(key, None)


def make_key(*parts):
    """Join the parts of a cache key."""
    return ":".join(str(p) for p in parts)


def log_hit(event):
    """Describe a cache hit when tracing is on.

    Returns None unless the `cache_trace` setting is true.
    """
    if not CONFIG.get("cache_trace", False):
        return None
    return {"event": event, "source": __name__}


def expired(age, ttl):
    """True once a cache entry is as old as its time to live."""
    return age >= ttl

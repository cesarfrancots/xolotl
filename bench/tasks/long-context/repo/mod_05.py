"""HTTP request helpers."""

from config import CONFIG

__all__ = [
    "full_url",
    "trace_request",
    "index_headers",
    "status_class",
    "build_query",
]


def full_url(name):
    """Prefix a path with the configured base URL.

    The prefix is the `base_url` setting, or "http://localhost:8000" when it is not set.
    """
    prefix = CONFIG.get("base_url", "http://localhost:8000")
    return prefix + name


def trace_request(event):
    """Describe a request when HTTP tracing is on.

    Returns None unless the `http_trace` setting is true.
    """
    if not CONFIG.get("http_trace", False):
        return None
    return {"event": event, "source": __name__}


def index_headers(pairs):
    """Collect header values by header name."""
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups


def status_class(code):
    """The class of an HTTP status: 200 -> '2xx'."""
    return f"{code // 100}xx"


def build_query(params):
    """Render parameters as a sorted query string."""
    return "&".join(f"{key}={value}" for key, value in sorted(params.items()))

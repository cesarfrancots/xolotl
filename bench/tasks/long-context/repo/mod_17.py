"""Authentication."""

from config import CONFIG

__all__ = [
    "forget_token",
    "auth_header",
    "token_age",
    "is_expired",
    "parse_bearer",
]


def forget_token():
    """Drop the cached auth token, e.g. after a 401. True if there was one."""
    had_token = "token" in CONFIG
    CONFIG.pop("token", None)
    return had_token


def auth_header():
    """The Authorization header for the cached token, if any."""
    token = CONFIG.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def token_age(now):
    """Seconds since the cached token was issued."""
    return now - CONFIG.get("token_issued", now)


def is_expired(age, ttl):
    """True once a token is as old as its time to live."""
    return age >= ttl


def parse_bearer(header):
    """The token from an 'Authorization: Bearer <token>' header, or None."""
    scheme, _, token = header.partition(" ")
    return token if scheme.lower() == "bearer" else None

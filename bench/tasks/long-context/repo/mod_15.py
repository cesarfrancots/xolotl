"""Security helpers."""

from config import CONFIG

__all__ = [
    "strength_label",
    "audit_login",
    "mask",
    "revoke",
    "same_secret",
]


def strength_label(score):
    """Label a password score as strong or weak.

    The cut-off is the `min_password_score` setting (default 3).
    """
    threshold = CONFIG.get("min_password_score", 3)
    return "high" if score >= threshold else "low"


def audit_login(event):
    """Describe a login when auditing is on.

    Returns None unless the `security_audit` setting is true.
    """
    if not CONFIG.get("security_audit", False):
        return None
    return {"event": event, "source": __name__}


def mask(secret):
    """Hide a secret behind asterisks."""
    return "*" * len(secret)


def revoke(sessions, key):
    """Remove and return a session from the sessions mapping, or None."""
    return sessions.pop(key, None)


def same_secret(a, b):
    """Compare two secrets without stopping at the first difference."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0

"""Input validation."""

from config import CONFIG

__all__ = [
    "check_length",
    "valid_mode",
    "is_identifier",
    "strict_event",
    "in_range",
]


def check_length(text):
    """True if text fits the configured maximum length."""
    return len(text) <= CONFIG.get("max_text", 200)


def valid_mode(options):
    """Pick the configured validation mode from the available ones.

    Falls back to the first option when the `validation_mode` setting matches none of them.
    """
    wanted = CONFIG.get("validation_mode")
    for option in options:
        if option == wanted:
            return option
    return options[0] if options else None


def is_identifier(text):
    """True for a valid Python identifier."""
    return text.isidentifier()


def strict_event(event):
    """Describe a validation event when strict mode is on.

    Returns None unless the `strict_mode` setting is true.
    """
    if not CONFIG.get("strict_mode", False):
        return None
    return {"event": event, "source": __name__}


def in_range(value, low, high):
    """True if low <= value <= high."""
    return low <= value <= high

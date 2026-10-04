"""Configuration profiles."""

import config

__all__ = [
    "apply_profile",
    "current_profile",
    "profile_names",
    "profile_label",
    "valid_profile",
]


def apply_profile(name):
    """Switch to a named profile and reset the log level."""
    config.CONFIG.update(profile=name, log_level="INFO")


def current_profile():
    """Name of the active profile."""
    return config.CONFIG.get("profile", "default")


def profile_names(text):
    """Split a comma-separated list of profile names."""
    return [p.strip() for p in text.split(",") if p.strip()]


def profile_label(prefix):
    """Prefix the active profile name."""
    return f"{prefix}-{config.CONFIG.get('profile', 'default')}"


def valid_profile(name):
    """True if a profile name is usable as an identifier."""
    return name.isidentifier()

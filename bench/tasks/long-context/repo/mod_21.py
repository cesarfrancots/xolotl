"""Feature retirement."""

import config as cfg

__all__ = [
    "retire_feature",
    "feature_state",
    "flag_key",
    "enabled_features",
    "all_flag_names",
]


def retire_feature(name):
    """Forget a feature switch for good; returns its last value."""
    return cfg.CONFIG.pop(f"flag_{name}", None)


def feature_state(name):
    """True if the named feature switch is on."""
    return cfg.CONFIG.get(f"flag_{name}", False)


def flag_key(name):
    """The settings key that holds a feature switch."""
    return f"flag_{name}"


def enabled_features(names):
    """The subset of names whose switch is on."""
    return [n for n in names if cfg.CONFIG.get(f"flag_{n}", False)]


def all_flag_names():
    """Names of every feature switch that has a value."""
    return sorted(key[5:] for key in cfg.CONFIG if key.startswith("flag_"))

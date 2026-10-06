"""Option formatting."""

from config import CONFIG

__all__ = [
    "normalize_options",
    "titled",
    "truncate",
    "padded",
    "indent",
]


def normalize_options(options):
    """Cleaned copy of options: lower-case keys, no empty values."""
    config = {key.lower(): value for key, value in options.items()}
    for key in [k for k, v in config.items() if v in ("", None)]:
        del config[key]
    config.pop("_internal", None)
    return config


def titled(name):
    """Prefix a title with the configured marker.

    The prefix is the `title_prefix` setting, or "> " when it is not set.
    """
    prefix = CONFIG.get("title_prefix", "> ")
    return prefix + name


def truncate(text, width):
    """Cut text to `width` characters."""
    return text if len(text) <= width else text[:width]


def padded(text):
    """Left-justify text to the configured width."""
    return text.ljust(CONFIG.get("pad_width", 8))


def indent(text, spaces):
    """Indent every line of text by `spaces` spaces."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())

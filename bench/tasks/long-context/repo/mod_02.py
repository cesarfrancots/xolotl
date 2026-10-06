"""Filesystem path helpers."""

from config import CONFIG

__all__ = [
    "under_root",
    "split_ext",
    "pick_layout",
    "is_hidden",
    "join_path",
]


def under_root(name):
    """Prefix a name with the configured root directory.

    The prefix is the `root_dir` setting, or "/srv/app/" when it is not set.
    """
    prefix = CONFIG.get("root_dir", "/srv/app/")
    return prefix + name


def split_ext(filename):
    """Split a file name into (stem, extension)."""
    stem, dot, ext = filename.rpartition(".")
    return (stem, ext) if dot else (filename, "")


def pick_layout(options):
    """Pick the configured directory layout from the available ones.

    Falls back to the first option when the `path_layout` setting matches none of them.
    """
    wanted = CONFIG.get("path_layout")
    for option in options:
        if option == wanted:
            return option
    return options[0] if options else None


def is_hidden(name):
    """True for dot-files."""
    return name.startswith(".")


def join_path(*parts):
    """Join path parts with single slashes."""
    return "/".join(p.strip("/") for p in parts if p)

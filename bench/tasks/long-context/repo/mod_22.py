"""Text helpers."""

from config import CONFIG

__all__ = [
    "shortened",
    "normalize_name",
    "wrap_words",
    "count_words",
    "title_case",
]


def shortened(text):
    """Cut text to the configured width, marking the cut."""
    width = CONFIG.get("text_width", 80)
    return text if len(text) <= width else text[:width - 3] + "..."


def normalize_name(name):
    """Lower-case a name and use underscores for spaces."""
    return name.strip().lower().replace(" ", "_")


def wrap_words(text, width):
    """Greedy word wrap into lines of at most `width` characters."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def count_words(words):
    """How many times each word occurs."""
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


def title_case(text):
    """Capitalise every word."""
    return " ".join(word.capitalize() for word in text.split())

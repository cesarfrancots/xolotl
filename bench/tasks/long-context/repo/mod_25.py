"""Reporting."""

from config import CONFIG

__all__ = [
    "report_title",
    "grade",
    "report_context",
    "bullet",
    "table",
]


def report_title(name):
    """Title line for a report, tagged with the environment."""
    return f"[{CONFIG.get('env', 'dev')}] {name}"


def grade(score):
    """Label a score as a pass or a fail.

    The cut-off is the `pass_mark` setting (default 60).
    """
    threshold = CONFIG.get("pass_mark", 60)
    return "high" if score >= threshold else "low"


def report_context(extra):
    """Settings plus the given extra values."""
    merged = dict(CONFIG)
    merged.update(extra)
    return merged


def bullet(lines):
    """Render lines as a bulleted list."""
    return "\n".join(f"- {line}" for line in lines)


def table(rows):
    """Render rows as left-aligned columns."""
    widths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows)

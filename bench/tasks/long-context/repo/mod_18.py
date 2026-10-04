"""Export helpers."""

from config import CONFIG

__all__ = [
    "pick_format",
    "quote",
    "public_view",
    "merge_rows",
    "rows_to_csv",
]


def pick_format(options):
    """Pick the configured export format from the available ones.

    Falls back to the first option when the `export_format` setting matches none of them.
    """
    wanted = CONFIG.get("export_format")
    for option in options:
        if option == wanted:
            return option
    return options[0] if options else None


def quote(field):
    """Quote one CSV field."""
    return '"' + field.replace('"', '""') + '"'


def public_view(*names):
    """Settings without the named keys."""
    view = dict(CONFIG)
    for name in names:
        view.pop(name, None)
    return view


def merge_rows(totals, more):
    """Add the entries of `more` into `totals` and return it."""
    totals.update(more)
    return totals


def rows_to_csv(rows):
    """Render rows of values as CSV text."""
    return "\n".join(",".join(quote(str(cell)) for cell in row) for row in rows)

def dedupe(items):
    """Return a list of items without duplicates, keeping the first occurrence of each, in order.

    Items are hashable values such as ints and strings.
    """
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out

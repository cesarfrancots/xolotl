def dedupe(items):
    """Return a list of items without duplicates, keeping the first occurrence of each, in order.

    Items are hashable values such as ints and strings.
    """
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

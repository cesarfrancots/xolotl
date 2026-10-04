def slugify(text: str) -> str:
    """Turn text into a URL slug.

    Apply these steps, in this order:
    1. Lowercase the text.
    2. Strip accents: normalize with NFKD and drop the combining marks.
    3. Replace each run of non-alphanumeric characters (anything that is not
       an ASCII letter a-z or digit 0-9) with a single "-".
    4. Trim "-" from both ends.

    Examples:
        "Hello, World!"    -> "hello-world"
        "  Crème Brûlée  " -> "creme-brulee"
    """
    raise NotImplementedError

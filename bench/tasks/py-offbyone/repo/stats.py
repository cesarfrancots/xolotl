def moving_average(xs, k):
    """Return the mean of each length-k window of xs, as a list of floats.

    Raises ValueError if k <= 0. Returns [] if k > len(xs).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return [sum(xs[i:i + k]) / k for i in range(len(xs) - k)]

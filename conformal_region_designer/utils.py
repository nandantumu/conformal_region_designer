import numpy


def conformalized_quantile(n, delta):
    """Given a number of samples and a desired coverage, return the conformalized quantile

    Args:
        n (int): Number of samples
        delta (float): Desired coverage

    Returns:
        float: Conformalized quantile
    """
    assert 0 < delta < 1
    return ((n + 1.0) * (delta)) / float(n)

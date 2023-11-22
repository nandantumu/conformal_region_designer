import numpy as np


def conformalized_quantile(n, delta):
    """Given a number of samples and a desired coverage, return the conformalized quantile

    Args:
        n (int): Number of samples
        delta (float): Desired coverage

    Returns:
        float: Conformalized quantile
    """
    assert 0 < delta < 1
    return np.ceil((n + 1.0) * (delta)) / float(n)


def shuffle_split_testing(pcr: "ConformalRegion", Z_eval: np.ndarray, num_splits: int, test_size: float, random_seed: int):
    """Split the evaluation set into multiple splits for testing

    Args:
        pcr (ConformalRegion): Conformal region
        Z_eval (np.ndarray): Evaluation set
        num_splits (int): Number of splits
        test_size (float): Size of the test set
        random_seed (int): Random seed

    Returns:
        list: List of tuples of (calibration set, test set)
    """
    from tqdm.auto import tqdm
    from sklearn.model_selection import ShuffleSplit
    rs = ShuffleSplit(n_splits=num_splits, test_size=test_size, random_state=random_seed)
    rs.get_n_splits(Z_eval)
    coverages = []
    with tqdm(total=rs.get_n_splits(Z_eval)) as pbar:
        for cal_index, test_index in rs.split(Z_eval):
            Z_cal = Z_eval[cal_index]
            Z_test = Z_eval[test_index]
            pcr.conformalize(Z_cal)
            # print(pcr.shapes[0].hyp_b)
            scores = pcr.calculate_scores(Z_test)
            coverage = np.sum(scores < 0) / len(scores)
            coverages.append(coverage)
            pbar.update(1)
            pbar.set_description(f"Coverage: {coverage:.0%}")
    return coverages
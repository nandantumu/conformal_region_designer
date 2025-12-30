from numpy.random.tests.test_smoke import RNG
from sklearn.cluster.tests.test_k_means import n_samples
import time
import numpy as np
from sklearn.neighbors import KernelDensity
from typing import Union, List
import jax.scipy as jsp
import jax.numpy as jnp
import jax

from .core import DensityEstimator


class KDE(DensityEstimator):
    def __init__(
        self, bandwidth="scott", kernel="gaussian", grid_size=100, bw_factor=1.0
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
        self.grid_size = grid_size
        self.bw_factor = bw_factor

    def fit(self, X):
        """
        Fit the density estimator to the data and save the range of the data

        Args:
            X (np.ndarray): Data to fit the density estimator to, shape (n_samples, n_features)
        """
        if self.bandwidth == "scott":
            self.bandwidth = X.shape[0] ** (-1 / (X.shape[1] + 4))
            self.bandwidth *= self.bw_factor
        elif self.bandwidth == "silverman":
            self.bandwidth = (X.shape[0] * (X.shape[1] + 2) / 4) ** (
                -1 / (X.shape[1] + 4)
            )
            self.bandwidth *= self.bw_factor
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(X)

        self.iqr = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
        # Add buffer to min/max, due to bandwidth of KDE
        self.min = np.min(X, axis=0) - 1.0 * self.iqr
        self.max = np.max(X, axis=0) + 1.0 * self.iqr

    def score_samples(self, X):
        return self.kde.score_samples(X)

    def generate_points(self, delta) -> np.ndarray:
        """
        Generate points from the density estimator that have sum of weight at least delta
        """
        # First generate a grid of points that covers the range of the data, with grid_size points in each dimension
        grid = np.meshgrid(
            *[
                np.linspace(self.min[i], self.max[i], self.grid_size)
                for i in range(len(self.min))
            ]
        )
        grid = np.array(grid).reshape(len(self.min), -1).T
        # Calculate the area of each grid cell
        grid_area = np.prod((self.max - self.min) / self.grid_size)
        # Calculate the density at each grid point
        grid_density = self.kde.score_samples(grid)
        # Calculate the weight of each grid point
        grid_weight = np.log(grid_area.astype(np.float64)) + grid_density.astype(
            np.float64
        )
        # Sort the grid points by weight from high to low
        sorted_indices = np.argsort(grid_weight)[::-1]
        # Reorder the grid and grid weight from increasing to decreasing density
        grid = grid[sorted_indices]
        grid_weight = grid_weight[sorted_indices]
        # Calculate the cumulative weight
        cum_weight = np.cumsum(np.exp(grid_weight))
        print(f"Total Weight Sum: {cum_weight[-1]}")
        # We may run into an issue, where the cumulative weight is not 1.0 due to numerical issues
        # We can fix this by renormalizing the cumulative weight
        cum_weight /= cum_weight[-1]
        # Find the first grid point with weight at least delta
        idx = np.argmax(cum_weight >= delta)
        # Return the grid points with weight at least delta
        return grid[: idx + 1]


class MCKDE(DensityEstimator):
    """
    This class provides a sampling based approximation of the high density regions by sampling from the KDE.
    """

    def __init__(
        self, bandwidth="scott", kernel="gaussian", sample_size=2_500, bw_factor=1.0
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
        self.bw_factor = bw_factor
        self.sample_size = sample_size

    def fit(self, X: np.ndarray):
        """
        Fit the density estimator to the data and save the range of the data

        Args:
            X (np.ndarray): Data to fit the density estimator to, shape (n_samples, n_features)
        """
        if self.bandwidth == "scott":
            self.bandwidth = X.shape[0] ** (-1 / (X.shape[1] + 4))
            self.bandwidth *= self.bw_factor
        elif self.bandwidth == "silverman":
            self.bandwidth = (X.shape[0] * (X.shape[1] + 2) / 4) ** (
                -1 / (X.shape[1] + 4)
            )
            self.bandwidth *= self.bw_factor
        self.kde = jsp.stats.gaussian_kde(dataset=X.T)
        KEY = jax.random.PRNGKey(seed=0)
        curr_key, KEY = jax.random.split(KEY)
        self.X_sample = self.kde.resample(curr_key, (self.sample_size,))

        self.iqr = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
        # Add buffer to min/max, due to bandwidth of KDE
        self.min = np.min(X, axis=0) - 1.0 * self.iqr
        self.max = np.max(X, axis=0) + 1.0 * self.iqr

    def generate_points(self, delta) -> np.ndarray:
        # We sample self.sample_size points from the KDE
        sample_scores = self.kde.evaluate(self.X_sample)
        threshold = jnp.quantile(sample_scores, delta)
        samples = self.X_sample.T[sample_scores >= threshold]
        return np.asarray(samples)


class NoOpDensityEstimator(DensityEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X):
        self.data = X

    def generate_points(self, delta) -> np.ndarray:
        return self.data

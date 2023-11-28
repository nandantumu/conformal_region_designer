import time
import numpy as np
from sklearn.neighbors import KernelDensity
from typing import Union, List

from .core import DensityEstimator


class KDE(DensityEstimator):
    def __init__(self, bandwidth="scott", kernel="gaussian", grid_size=100, bw_factor=1.0):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
        self.grid_size = grid_size
        self.bw_factor = bw_factor

    def fit(self, X, delta):
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

        self.delta = delta

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


# class KDE_TimeSeries(KDE):
#     def __init__(self, 
#                  bandwidth: Union[str, float, List[Union[str, float]]]="scott", 
#                  kernel: Union[str, List[str]] = "gaussian", 
#                  grid_size: Union[int, List[int]] = 100, 
#                  bw_factor: Union[int, List[int]] = 1.0):
#         super().__init__(bandwidth, kernel, grid_size, bw_factor)

#     def _timestep_parameters(self, X):
#         """
#         Convert single parameters to lists of parameters for each timestep
#         """
#         if isinstance(self.bandwidth, float) or isinstance(self.bandwidth, str):
#             self.bandwidth = [self.bandwidth] * X.shape[1]
#         if isinstance(self.kernel, str):
#             self.kernel = [self.kernel] * X.shape[1]
#         if isinstance(self.grid_size, int):
#             self.grid_size = [self.grid_size] * X.shape[1]
#         if isinstance(self.bw_factor, int):
#             self.bw_factor = [self.bw_factor] * X.shape[1]

        
#     def fit(self, X, delta):
#         """
#         Fit the density estimator to the data and save the range of the data
#         This version of fit fits a KDE to each timestep of the data. 
#         The weight of a point is the average of the weights in each timestep


#         Args:
#             X (np.ndarray): Data to fit the density estimator to, shape (n_samples, n_features)
#         """
#         self._timestep_parameters(X)
#         self.kde = [None for _ in range(X.shape[1])]
#         self.min = [None for _ in range(X.shape[1])]
#         self.max = [None for _ in range(X.shape[1])]

#         for t in range(X.shape[1]):
#             if self.bandwidth[t] == "scott":
#                 self.bandwidth[t] = X.shape[0] ** (-1 / (X.shape[1] + 4))
#                 self.bandwidth[t] *= self.bw_factor[t]
#             elif self.bandwidth[t] == "silverman":
#                 self.bandwidth[t] = (X.shape[0] * (X.shape[1] + 2) / 4) ** (
#                         -1 / (X.shape[1] + 4)
#                 )
#                 self.bandwidth[t] *= self.bw_factor[t]
#             self.kde[t] = KernelDensity(bandwidth=self.bandwidth[t], kernel=self.kernel[t])
#             self.kde[t].fit(X[:, t].reshape(-1, 1))

#             self.iqr = np.quantile(X[:, t], 0.75) - np.quantile(X[:, t], 0.25)
#             # Add buffer to min/max, due to bandwidth of KDE
#             self.min[t] = np.min(X[:, t]) - 1.0 * self.iqr
#             self.max[t] = np.max(X[:, t]) + 1.0 * self.iqr

#         self.delta = delta

#     def score_samples(self, X):
#         """
#         Calculate the score samples for each timestep
#         """
#         scores = np.zeros((X.shape[0], X.shape[1]))
#         for t in range(X.shape[1]):
#             scores[:, t] = self.kde[t].score_samples(X[:, t].reshape(-1, 1))
#         # Average the scores across timesteps
#         scores = np.mean(scores, axis=1)
#         return scores

#     def generate_points(self, delta) -> np.ndarray:
#         """
#         Generate points from the density estimator that have sum of weight at least delta
#         """
#         # We need to generate grid boundaries for each timestep independently, since the range of each timestep may be different
#         # Then we need to combine the grid points back together
#         timesteps = len(self.min)
#         grid = [None for _ in range(timesteps)]
#         grid_area = [None for _ in range(timesteps)]
#         grid_density = [None for _ in range(timesteps)]
#         grid_weight = [None for _ in range(timesteps)]
#         sorted_indices = [None for _ in range(timesteps)]
#         cum_weight = [None for _ in range(timesteps)]
#         # Generate the grid for each timestep
#         for t in range(timesteps):
#             grid[t] = np.linspace(self.min[t], self.max[t], self.grid_size[t])
#             grid_area[t] = (self.max[t] - self.min[t]) / self.grid_size[t]
#             grid_density[t] = self.kde[t].score_samples(grid[t].reshape(-1, 1))
#             grid_weight[t] = np.log(grid_area[t].astype(np.float64)) + grid_density[t].astype(np.float64)
#             sorted_indices[t] = np.argsort(grid_weight[t])[::-1]
#             grid[t] = grid[t][sorted_indices[t]]
#             grid_weight[t] = grid_weight[t][sorted_indices[t]]
#             cum_weight[t] = np.cumsum(np.exp(grid_weight[t]))
#             cum_weight[t] /= timesteps

#         # We want to find the global threshold such that the sum of weights is at least delta



class NoOpDensityEstimator(DensityEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, delta):
        self.data = X

    def generate_points(self, delta) -> np.ndarray:
        return self.data
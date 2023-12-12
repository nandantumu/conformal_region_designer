"""
This package implements parametric search algorithms for conformal prediction regions.


"""
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class DensityEstimator(ABC):
    """This class is an ABC for a density estimator"""

    @abstractmethod
    def fit(self, X):
        """Fit the density estimator to the data

        Args:
            X (np.ndarray): Data to fit the density estimator to
        """
        pass

    @abstractmethod
    def generate_points(self, delta) -> np.ndarray:
        """
        Generate points from the density estimator that have weight at least delta
        {x: p(x)>delta}
        """
        pass


class Clustering(ABC):
    """This class is an ABC for a clustering algorithm"""

    @abstractmethod
    def fit(self, X):
        """Fit clusters to data"""

    @abstractmethod
    def generate_clustered_points(self) -> List[np.ndarray]:
        """Generate points that are part of a cluster"""
        pass


class ShapeTemplate(ABC):
    """Given a set of points, generate a convex hull around those points"""

    def fit_shape(self, Z_cal_one):
        """Create convex shape to encompass all of the points in X"""
        pass

    def score_points(self, Z):
        """Given a set of points, score them"""
        pass

    def conformalize(self, delta, Z_cal_two):
        """Given a convex hull, return a convex set that has weight at least delta
        {x: p(x)>delta}
        """
        pass

    def adjust_shape(self, score_margin):
        """Adjust the shape (inflate or deflate) based on the score margin"""
        pass

    def volume(self):
        """Return the volume of the shape"""
        raise NotImplementedError

    def plot(self, ax, offset_coords=None):
        """Plot the shape"""
        pass

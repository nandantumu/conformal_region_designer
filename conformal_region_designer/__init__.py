"""
This package implements parametric search algorithms for conformal prediction regions.


"""
from abc import ABC, abstractmethod
from typing import List, Union
from numpy import np

class DensityEstimator(ABC):
    """This class is an ABC for a density estimator
    """

    @abstractmethod
    def fit(self, X):
        """Fit the density estimator to the data

        Args:
            X (np.ndarray): Data to fit the density estimator to
        """
        pass

    @abstractmethod
    def generate_points(self, delta):
        """
        Generate points from the density estimator that have weight at least delta
        {x: p(x)>delta}
        """
        pass

class Clustering(ABC):
    """This class is an ABC for a clustering algorithm
    """

    @abstractmethod
    def fit(self, X):
        """ Fit clusters to data"""

    @abstractmethod
    def generate_clustered_points(self) -> List[np.ndarray]:
        """ Generate points that are part of a cluster
        """
        pass

class SingleConvexifier(ABC):
    """Given a set of points, generate a convex hull around those points
    """
    def generate_convex_shape(self, X):
        """Create convex shape to encompass all of the points in X
        """
        pass

    def conformalize(self, CHull, delta, new_data, score_function):
        """Given a convex hull, return a convex set that has weight at least delta
        {x: p(x)>delta}
        """
        pass


class MultimodalConvexifier(ABC):
    """Given a set of points and a set of clusters, generate a set of convex hulls around those points
    """
    
    def generate_convex_shapes(self, clusters):
        """Create convex shapes to encompass all of the points in X
        """
        pass

    def conformalize(self, CHulls, delta, new_data, score_function):
        """Given a convex hull, return a convex set that has weight at least delta
        {x: p(x)>delta}
        """
        pass

class ScoreFunction(ABC):

    def score_points(self, X):
        """Given a set of points, score them
        """
        pass
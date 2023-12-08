import numpy as np

from ..core import ShapeTemplate
from ..utils import conformalized_quantile


class HyperrectangleTemplate(ShapeTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.min = None
        self.max = None

    def fit_shape(self, X):
        """Create a hyperrectangle that encompasses all of the points in X"""
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.aspect_ratio = (self.max - self.min) / np.max(self.max - self.min)

    def score_points(self, X):
        """The score of a point is the distance to the nearest edge of the hyperrectangle"""
        score = np.maximum(X - self.max, self.min - X)*self.aspect_ratio
        return score.max(axis=1)

    def conformalize(self, delta, calibration_data):
        """
        Find the score of the calibration data
        """
        scores = self.score_points(calibration_data)
        inflation = np.quantile(scores, conformalized_quantile(len(scores), delta))
        self.min -= inflation
        self.max += inflation
        return self.min, self.max

    def adjust_shape(self, score_margin):
        self.min -= score_margin
        self.max += score_margin

    def volume(self):
        return np.prod(self.max - self.min)

    def plot(self, ax, offset_coords=None, **kwargs):
        """
        Plot the hyperrectangle in 2d and 3d
        """
        pltargs = {"color": "black"}
        pltargs.update(kwargs)
        if offset_coords is None:
            offset_coords = np.zeros(len(self.min))
        if len(self.min) == 2:
            ax.plot(
                np.array([self.min[0], self.min[0], self.max[0], self.max[0], self.min[0]])+offset_coords[0],
                np.array([self.min[1], self.max[1], self.max[1], self.min[1], self.min[1]])+offset_coords[1],
                **pltargs
            )
        elif len(self.min) == 3:
            ax.plot(
                np.array([self.min[0], self.min[0], self.max[0], self.max[0], self.min[0]])+offset_coords[0],
                np.array([self.min[1], self.max[1], self.max[1], self.min[1], self.min[1]])+offset_coords[1],
                np.array([self.min[2], self.min[2], self.min[2], self.min[2], self.min[2]])+offset_coords[2],
                **pltargs
            )
            ax.plot(
                np.array([self.min[0], self.min[0], self.max[0], self.max[0], self.min[0]])+offset_coords[0],
                np.array([self.min[1], self.max[1], self.max[1], self.min[1], self.min[1]])+offset_coords[1],
                np.array([self.max[2], self.max[2], self.max[2], self.max[2], self.max[2]])+offset_coords[2],
                **pltargs,
            )
            ax.plot(
                np.array([self.min[0], self.min[0]])+offset_coords[0],
                np.array([self.min[1], self.min[1]])+offset_coords[1],
                np.array([self.min[2], self.max[2]])+offset_coords[2],
                **pltargs
            )
            ax.plot(
                np.array([self.min[0], self.min[0]])+offset_coords[0],
                np.array([self.max[1], self.max[1]])+offset_coords[1],
                np.array([self.min[2], self.max[2]])+offset_coords[2],
                **pltargs
            )
            ax.plot(
                np.array([self.max[0], self.max[0]])+offset_coords[0],
                np.array([self.min[1], self.min[1]])+offset_coords[1],
                np.array([self.min[2], self.max[2]])+offset_coords[2],
                **pltargs
            )
            ax.plot(
                np.array([self.max[0], self.max[0]])+offset_coords[0],
                np.array([self.max[1], self.max[1]])+offset_coords[1],
                np.array([self.min[2], self.max[2]])+offset_coords[2],
                **pltargs
            )
        else:
            raise NotImplementedError(
                "Hyperrectangle plotting only implemented for 2d and 3d"
            )

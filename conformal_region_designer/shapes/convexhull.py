import numpy as np
from scipy.spatial import ConvexHull as CHull, HalfspaceIntersection as HSI

from ..core import ShapeTemplate
from ..utils import conformalized_quantile


class ConvexHullTemplate(ShapeTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.hull = None

    def fit_shape(self, X):
        """Create a ConvexHull that encompasses all of the points in X"""
        self.hull = CHull(X)
        self.hull.close()  # Free resources
        self.interior_point = np.mean(X, axis=0)
        self.equations = self.hull.equations
        self.hyp_a = self.equations[:, :-1]
        self.hyp_b = self.equations[:, -1]
        self.hs = None

    def score_points(self, X):
        return np.max(X@self.hyp_a.T + self.hyp_b, axis=1)
    
    def conformalize(self, delta, calibration_data):
        """
        Find the score of the calibration data
        """
        scores = self.score_points(calibration_data)
        inflation = np.quantile(scores, conformalized_quantile(len(scores), delta))
        self.adjust_shape(inflation)
        
    def adjust_shape(self, score_margin):
        self.hyp_b -= score_margin
        self.hs = None
        self.hull = None

    def volume(self):
        """Compute the volume of the convex hull"""
        if self.hull is None:
            self._generate_cvxhull()
        return self.hull.volume
    
    def _generate_hsi(self):
        if self.hs is None:
            self.hs = HSI(np.hstack([self.hyp_a, self.hyp_b[:, None]]), self.interior_point)
            
    def _generate_cvxhull(self):
        self._generate_hsi()
        self.hull = CHull(self.hs.intersections)
        self.hull.close()

    def plot(self, ax, offset_coords=None, **kwargs):
        """Convert the halfspace equations to a halfspace intersection, and plot the vertices"""
        if offset_coords is None:
            offset_coords = np.zeros(self.hyp_a.shape[1])
        if self.hyp_a.shape[1] == 3:
            # Plot the convex hull in 3d
            raise NotImplementedError("3d plotting not implemented yet")
        elif self.hyp_a.shape[1] == 2:
            # Plot the convex hull in 2d
            if self.hs is None:
                self.hs = HSI(np.hstack([self.hyp_a, self.hyp_b[:, None]]), self.interior_point)
            vertices = self.hs.intersections
            # Sort these vertices to be clockwise
            centroid = np.mean(vertices, axis=0)
            angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
            vertices = vertices[np.argsort(angles)]
            # Add the first vertex to the end to close the polygon
            vertices = np.vstack([vertices, vertices[0]])
            # Plot the vertices
            ax.plot(vertices[:, 0]+offset_coords[0], vertices[:, 1]+offset_coords[1], color="black")

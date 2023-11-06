"""
This file contains an implementation of the overall orchestrator that creates regions.

"""
import numpy as np

from .core import Clustering, DensityEstimator, ShapeTemplate
from .utils import conformalized_quantile


class ConformalRegion:
    def __init__(
        self, de: DensityEstimator, cl: Clustering, st: type[ShapeTemplate], delta=0.95
    ) -> None:
        self.de = de
        self.cl = cl
        self.st = st
        self.delta = delta

    def fit(self, Z_train: np.ndarray):
        self.de.fit(Z_train)
        print("Generating density points")
        self.density_points = self.de.generate_points(self.delta)
        print("Fitting Clusters")
        self.cl.fit(self.density_points)
        self.clusters = self.cl.generate_clustered_points(self.density_points)
        print("Fitting Shapes")
        self.shapes = [self.st() for _ in range(len(self.clusters))]
        for shape, cluster in zip(self.shapes, self.clusters):
            shape.fit_shape(cluster)

    def conformalize(self, Z_cal: np.ndarray):
        conf_delta = conformalized_quantile(len(Z_cal), self.delta)
        scores = np.zeros((len(self.shapes), Z_cal.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_cal)
        real_scores = np.min(scores, axis=0)
        shape_idx = np.argmin(scores, axis=0)
        target_score = np.quantile(real_scores, conf_delta)
        for i, shape in enumerate(self.shapes):
            shape.adjust_shape(target_score)

    def calculate_scores(self, Z_test: np.ndarray):
        scores = np.zeros((len(self.shapes), Z_test.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_test)
        return np.min(scores, axis=0)

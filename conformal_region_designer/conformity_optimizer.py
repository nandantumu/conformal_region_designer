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
        # We need to compute a normalizing constant for each shape
        scores = np.zeros((len(self.shapes), Z_train.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_train)
        real_scores = np.min(scores, axis=0)
        shape_idx = np.argmin(scores, axis=0)
        # For each shape compute the max score of the points assigned to it
        self.normalizing_constant = np.zeros(len(self.shapes))
        for i in range(len(self.shapes)):
            self.normalizing_constant[i] = np.max(real_scores[shape_idx == i])

    def conformalize(self, Z_cal: np.ndarray):
        conf_delta = conformalized_quantile(len(Z_cal), self.delta)
        scores = np.zeros((len(self.shapes), Z_cal.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_cal)/self.normalizing_constant[i]
        real_scores = np.min(scores, axis=0)
        shape_idx = np.argmin(scores, axis=0)
        target_score = np.quantile(real_scores, conf_delta)
        print(f"Target score: {target_score}")
        for i, shape in enumerate(self.shapes):
            shape.adjust_shape(target_score*self.normalizing_constant[i])

    def calculate_scores(self, Z_test: np.ndarray):
        scores = np.zeros((len(self.shapes), Z_test.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_test)/self.normalizing_constant[i]
        return np.min(scores, axis=0)

"""
This file contains an implementation of the overall orchestrator that creates regions.

"""
from copy import deepcopy
from typing import Union
import numpy as np

from .core import Clustering, DensityEstimator, ShapeTemplate
from .utils import conformalized_quantile


class ConformalRegion:
    def __init__(
        self, 
        de: Union[DensityEstimator, str] = 'kde', 
        cl: Union[Clustering, str] = 'meanshift', 
        st: Union[type[ShapeTemplate], str] = 'hyperrectangle', 
        delta=0.95
    ) -> None:
        if isinstance(de, str):
            if de == 'kde':
                from .density_estimation import KDE
                de = KDE()
            else:
                raise ValueError(f"Unknown density estimator {de}")
        self.de = de
        if isinstance(cl, str):
            if cl == 'meanshift':
                from .clustering import MeanShiftClustering
                cl = MeanShiftClustering()
            else:
                raise ValueError(f"Unknown clustering algorithm {cl}")
        self.cl = cl
        if isinstance(st, str):
            if st == 'hyperrectangle':
                from .shapes import HyperrectangleTemplate
                st = HyperrectangleTemplate
            elif st == 'convexhull':
                from .shapes import ConvexHullTemplate
                st = ConvexHullTemplate
            elif st == 'ellipse':
                from .shapes import EllipsoidTemplate
                st = EllipsoidTemplate
            else:
                raise ValueError(f"Unknown shape template {st}")
        self.st = st
        self.delta = delta

    def fit(self, Z_train: np.ndarray, verbose: bool=False):
        self.de.fit(Z_train, self.delta)
        if verbose: print("Generating density points")
        self.density_points = self.de.generate_points(self.delta)
        if verbose: print("Fitting Clusters")
        self.cl.fit(self.density_points)
        self.clusters = self.cl.generate_clustered_points(self.density_points)
        if verbose: print("Fitting Shapes")
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
        self.normalizing_constant = np.ones(len(self.shapes))
        for i in range(len(self.shapes)):
            self.normalizing_constant[i] = np.quantile(real_scores[shape_idx == i], self.delta)    
        try:
            self.normalizing_constant += np.min(self.normalizing_constant) + 1.0
            self.normalizing_constant/= np.sum(self.normalizing_constant)
        except:
            self.normalizing_constant = np.ones(len(self.shapes))

    def conformalize(self, Z_cal: np.ndarray, debug=False):
        conf_delta = conformalized_quantile(len(Z_cal), self.delta)
        scores = np.zeros((len(self.shapes), Z_cal.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_cal)*self.normalizing_constant[i]
        real_scores = np.min(scores, axis=0)
        shape_idx = np.argmin(scores, axis=0)
        target_score = np.quantile(real_scores, conf_delta)
        if debug: print(f"Target score: {target_score}")
        self.adjust_shapes(target_score)

    def adjust_shapes(self, target_score: float):
        for i, shape in enumerate(self.shapes):
            shape.adjust_shape(target_score/self.normalizing_constant[i])

    def calculate_scores(self, Z_test: np.ndarray):
        scores = np.zeros((len(self.shapes), Z_test.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_test)*self.normalizing_constant[i]
        return np.min(scores, axis=0)
    
    def volume(self):
        return np.sum([shape.volume() for shape in self.shapes])


class ConformalRegionTimeSeries(ConformalRegion):
    """
    This class is a wrapper around ConformalRegion that handles time series data.
    It does so by fitting a ConformalRegion to each timestep of the data, and then
    using a normalizing constant over each of the timesteps to achieve timeseries regions
    """
    def __init__(
        self, 
        timesteps: int,
        de: Union[DensityEstimator, str] = 'kde', 
        cl: Union[Clustering, str] = 'meanshift', 
        st: Union[type[ShapeTemplate], str] = 'hyperrectangle', 
        delta=0.95
    ) -> None:
        self.cregions = [ConformalRegion(de, cl, st, delta) for _ in range(timesteps)]
        self.timesteps = timesteps
        self.delta = delta

    def fit(self, Z_train: np.ndarray, verbose: bool=False):
        for i, cregion in enumerate(self.cregions):
            if verbose: print(f"Fitting timestep {i}")
            cregion.fit(Z_train[:, i].reshape(Z_train.shape[0], -1), verbose=verbose)
        scores = np.zeros((Z_train.shape[0], self.timesteps))
        for i, cregion in enumerate(self.cregions):
            scores[:, i] = cregion.calculate_scores(Z_train[:, i].reshape(Z_train.shape[0], -1))
        real_scores = np.min(scores, axis=1)
        ts_idx = np.argmin(scores, axis=1)
        self.normalizing_constant = np.ones(self.timesteps)
        for i in range(self.timesteps):
            self.normalizing_constant[i] = np.quantile(real_scores[ts_idx == i], self.delta)
        try:
            self.normalizing_constant += np.min(self.normalizing_constant) + 1.0
            self.normalizing_constant/= np.sum(self.normalizing_constant)
        except:
            self.normalizing_constant = np.ones(len(self.shapes))

    def conformalize(self, Z_cal: np.ndarray, debug=False):
        conf_delta = conformalized_quantile(len(Z_cal), self.delta)
        scores = np.zeros((Z_cal.shape[0], self.timesteps))
        for i, cregion in enumerate(self.cregions):
            scores[:, i] = cregion.calculate_scores(Z_cal[:, i].reshape(Z_cal.shape[0], -1))*self.normalizing_constant[i]
        real_scores = np.max(scores, axis=1)
        ts_idx = np.argmax(scores, axis=1)
        target_score = np.quantile(real_scores, conf_delta)
        if debug: print(f"Target score: {target_score}")
        for i, cregion in enumerate(self.cregions):
            cregion.adjust_shapes(target_score/self.normalizing_constant[i])
    
    def calculate_scores(self, Z_test: np.ndarray):
        scores = np.zeros((Z_test.shape[0], self.timesteps))
        for i, cregion in enumerate(self.cregions):
            scores[:, i] = cregion.calculate_scores(Z_test[:, i].reshape(Z_test.shape[0], -1))*self.normalizing_constant[i]
        return np.max(scores, axis=1)

    def volume(self):
        return np.sum([cregion.volume() for cregion in self.cregions])
    
    @property
    def shapes(self):
        return sum([cregion.shapes for cregion in self.cregions])
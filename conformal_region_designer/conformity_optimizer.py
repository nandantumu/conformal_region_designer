"""
This file contains an implementation of the overall orchestrator that creates regions.

"""
from copy import deepcopy
from typing import Union
import numpy as np
from time import time

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

    def fit(self, Z_cal_one: np.ndarray, verbose: bool=False):
        de_start = time()
        self.de.fit(Z_cal_one, self.delta)
        if verbose: print("Generating density points")
        self.density_points = self.de.generate_points(self.delta)
        de_end = time()
        if verbose: print("Fitting Clusters")
        self.cl.fit(self.density_points)
        self.clusters = self.cl.generate_clustered_points(self.density_points)
        cl_end = time()
        if verbose: print("Fitting Shapes")
        self.shapes = [self.st() for _ in range(len(self.clusters))]
        for shape, cluster in zip(self.shapes, self.clusters):
            shape.fit_shape(cluster)
        st_end = time()
        # We need to compute a normalizing constant for each shape
        scores = np.zeros((len(self.shapes), Z_cal_one.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_cal_one)
        real_scores = np.min(scores, axis=0)
        shape_idx = np.argmin(scores, axis=0)
        # For each shape compute the max score of the points assigned to it
        self.normalizing_constant = 1 /(1e-8 + np.quantile(scores, self.delta, axis=1) - np.min(scores, axis=1))
        # self.normalizing_constant = self.normalizing_constant/np.sum(self.normalizing_constant)
        # np.ones(len(self.shapes))
        # self.cal_bound = np.ones(len(self.shapes))
        # self.cal_min = np.zeros(len(self.shapes))
        # for i in range(len(self.shapes)):
        #     self.cal_bound[i] = np.quantile(real_scores[shape_idx == i], self.delta)
        #     self.cal_min[i] = np.min(real_scores[shape_idx == i])
        # try:
        #     self.normalizing_constant = 1/(self.cal_bound - self.cal_min)
        # except:
        #     self.normalizing_constant = np.ones(len(self.shapes))
        score_end = time()
        self.de_time = de_end - de_start
        self.cl_time = cl_end - de_end
        self.st_time = st_end - cl_end
        self.score_time = score_end - st_end

    def conformalize(self, Z_cal_two: np.ndarray, debug=False):
        start_time = time()
        conf_delta = conformalized_quantile(len(Z_cal_two), self.delta)
        scores = np.zeros((len(self.shapes), Z_cal_two.shape[0]))
        for i, shape in enumerate(self.shapes):
            scores[i] = shape.score_points(Z_cal_two)*self.normalizing_constant[i]
        real_scores = np.min(scores, axis=0)
        shape_idx = np.argmin(scores, axis=0)
        target_score = np.quantile(real_scores, conf_delta)
        if debug: print(f"Target score: {target_score}")
        self.adjust_shapes(target_score)
        end_time = time()
        self.conformalize_time = end_time - start_time
    
    def print_times(self):
        """Print the times for each step of the process"""
        print(f"DE time             : {self.de_time}")
        print(f"CL time             : {self.cl_time}")
        print(f"ST time             : {self.st_time}")
        print(f"Score time          : {self.score_time}")
        try:
            print(f"Conformalize time   : {self.conformalize_time}")
            print(f"Total time          : {self.de_time + self.cl_time + self.st_time + self.score_time + self.conformalize_time}")
        except:
            print(f"Total time          : {self.de_time + self.cl_time + self.st_time + self.score_time}")

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

    def fit(self, Z_cal_one: np.ndarray, verbose: bool=False):
        for i, cregion in enumerate(self.cregions):
            if verbose: print(f"Fitting timestep {i}")
            cregion.fit(Z_cal_one[:, i].reshape(Z_cal_one.shape[0], -1), verbose=verbose)
        scores = np.zeros((Z_cal_one.shape[0], self.timesteps))
        for i, cregion in enumerate(self.cregions):
            scores[:, i] = cregion.calculate_scores(Z_cal_one[:, i].reshape(Z_cal_one.shape[0], -1))
        real_scores = np.min(scores, axis=1)
        ts_idx = np.argmin(scores, axis=1)
        self.time_normalizing_constant = 1/(1e-8 + np.quantile(scores, self.delta, axis=1) - np.min(scores, axis=1))
        # self.time_normalizing_constant = self.time_normalizing_constant/np.sum(self.time_normalizing_constant)
        
        # cal_one_bounds = np.ones(self.timesteps)
        # cal_one_mins = np.zeros(self.timesteps)
        # for i in range(self.timesteps):
        #     try:
        #         cal_one_bounds[i] = np.quantile(real_scores[ts_idx == i], self.delta)
        #         cal_one_mins[i] = np.min(real_scores[ts_idx == i])
        #     except IndexError:
        #         cal_one_bounds[i] = 1.#-np.inf
        #         cal_one_mins[i] = 0.
        # try:
        #     # Set the np.inf values to the minimum value + 1
        #     # self.normalizing_constant[self.normalizing_constant == -np.inf] = np.min(self.normalizing_constant[self.normalizing_constant != -np.inf])
        #     self.time_normalizing_constant = 1/(cal_one_bounds - cal_one_mins)
        #     # self.time_normalizing_constant -= np.min(self.time_normalizing_constant) - 1.0
        #     # self.time_normalizing_constant = 1/self.time_normalizing_constant
        #     # We now are in the range [0, 1]
        #     # We can now compute a second step approximation of the normalizing constant
        #     # adjusted_scores = np.min(scores*self.time_normalizing_constant, axis=1)
        #     # adjusted_score_index = np.argmin(scores*self.time_normalizing_constant, axis=1)
        #     # adjusted_bounds = np.array([np.quantile(adjusted_scores[adjusted_score_index == i], self.delta) for i in range(self.timesteps)])
        #     # calibrated_hypothesis = cal_one_bounds * self.time_normalizing_constant
        #     # self.time_normalizing_constant *= 1/calibrated_hypothesis

        # except:
        #     self.time_normalizing_constant = np.ones(len(self.shapes))

    def conformalize(self, Z_cal_two: np.ndarray, debug=False):
        conf_delta = conformalized_quantile(len(Z_cal_two), self.delta)
        scores = np.zeros((Z_cal_two.shape[0], self.timesteps))
        for i, cregion in enumerate(self.cregions):
            scores[:, i] = cregion.calculate_scores(Z_cal_two[:, i].reshape(Z_cal_two.shape[0], -1))*self.time_normalizing_constant[i]
        real_scores = np.max(scores, axis=1)
        ts_idx = np.argmax(scores, axis=1)
        target_score = np.quantile(real_scores, conf_delta)
        if debug: print(f"Target score: {target_score}")
        for i, cregion in enumerate(self.cregions):
            cregion.adjust_shapes(target_score/self.time_normalizing_constant[i])
    
    def calculate_scores(self, Z_test: np.ndarray):
        scores = np.zeros((Z_test.shape[0], self.timesteps))
        for i, cregion in enumerate(self.cregions):
            scores[:, i] = cregion.calculate_scores(Z_test[:, i].reshape(Z_test.shape[0], -1))*self.time_normalizing_constant[i]
        return np.max(scores, axis=1)

    def volume(self):
        return np.sum([cregion.volume() for cregion in self.cregions])
    
    @property
    def shapes(self):
        import itertools
        return list(itertools.chain.from_iterable([cregion.shapes for cregion in self.cregions]))
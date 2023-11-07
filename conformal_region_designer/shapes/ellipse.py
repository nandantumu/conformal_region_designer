import math
import pickle
import random
import time

import cma
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np

from ..core import ShapeTemplate

PLOT_VALIDATION_TRACES = True
NUM_VALID_TO_PLOT = 100


def computeCPEllipseMatrixManyEllipse(residuals, Q_matrices, ellipse_centers, delta):
    R_vals = [
        min(
            [
                np.matmul(
                    np.matmul(residuals[i] - ellipse_centers[j], Q_matrices[j]),
                    residuals[i] - ellipse_centers[j],
                )
                for j in range(len(Q_matrices))
            ]
        )
        for i in range(residuals.shape[0])
    ]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret].item()


def callCMAESMatrixManyEllipse(residuals, delta, num_ellipse, ellipse_centers=None):
    ## add in cma example
    # residuals = np.array(residuals)
    dims = residuals.shape[1]
    args_cma = [residuals, delta, dims, num_ellipse, ellipse_centers]

    ## TODO: figure out how to represent x_0, then write new objection_func_cma_es_matrix_ellipse() function (this part should be simple)
    # can we just have x_0 be a matrix of the right dimension? Wouldn't the sampling not be correct?
    # can we use the sqrt(Q) as the thing being sampled?

    if ellipse_centers is None:
        num_ellipse_params = dims**2 + dims
        pre_defined_centers = False
    else:
        num_ellipse_params = dims**2
        pre_defined_centers = True
    input_size = num_ellipse * num_ellipse_params
    x_0 = np.zeros(input_size)

    ## add more informative guess
    for i in range(num_ellipse):
        a = np.eye(dims).reshape(dims**2)
        x_0[i * num_ellipse_params : i * num_ellipse_params + dims**2] = a

    sigma0 = 1  # [0.25,1,0.25] # the std of the search space. Should be about 1/4 of size of search space. This is a complete guess

    start_time = time.time()
    x, es = cma.fmin2(
        objective_func_cma_es_ellipse_arbitrary_dim_multi_ellipse,
        x_0,
        sigma0,
        args=args_cma,
    )
    end_time = time.time()

    Q_matrices = []
    if not pre_defined_centers:
        ellipse_centers = []
    for i in range(num_ellipse):
        x_temp = x[i * num_ellipse_params : i * num_ellipse_params + dims**2]
        x_temp = x_temp.reshape(
            (int(math.sqrt(x_temp.size)), int(math.sqrt(x_temp.size)))
        )
        if not pre_defined_centers:
            center_temp = x[
                i * num_ellipse_params + dims**2 : (i + 1) * num_ellipse_params
            ]
            ellipse_centers.append(center_temp)
        Q = np.matmul(x_temp.T, x_temp)
        Q_matrices.append(Q)

    # print("cmaes soln: " + str(x))
    print(Q_matrices)
    print(ellipse_centers)
    print("Soln time: " + str(end_time - start_time))

    return Q_matrices, ellipse_centers


def objective_func_cma_es_ellipse_arbitrary_dim_multi_ellipse(x, *args):
    residuals = args[0]
    delta = args[1]
    dims = args[2]
    num_ellipse = args[3]
    if args[4] is None:
        ellipse_centers = None
        num_ellipse_params = (
            dims**2 + dims
        )  # first term for the matrix, second for the offset
        pre_defined_centers = False
    else:
        ellipse_centers = args[4]
        num_ellipse_params = dims**2
        pre_defined_centers = True

    assert num_ellipse_params * num_ellipse == len(x)  # double check my algebra

    Q_matrices = []
    if not pre_defined_centers:
        ellipse_centers = []
    for i in range(num_ellipse):
        x_temp = x[i * num_ellipse_params : i * num_ellipse_params + dims**2]
        x_temp = x_temp.reshape(
            (int(math.sqrt(x_temp.size)), int(math.sqrt(x_temp.size)))
        )

        if not pre_defined_centers:
            center_temp = x[
                i * num_ellipse_params + dims**2 : (i + 1) * num_ellipse_params
            ]
            ellipse_centers.append(center_temp)
        Q = np.matmul(x_temp.T, x_temp)

        if np.linalg.matrix_rank(Q) != Q.shape[0]:  ## Q not full rank
            return np.NaN
        if not (
            np.allclose(Q, Q.T, rtol=1e-05, atol=1e-08)
        ):  ## Q not symmetric (should not be possible for this to happen given how Q is constructed)
            return np.NaN
        Q_matrices.append(Q)

    D_cp = computeCPEllipseMatrixManyEllipse(
        residuals, Q_matrices, ellipse_centers, delta
    )
    ## what should the objective function be? sum of volumes? What if the ellipses overlap?

    ## compute area of ellpise given matrices are Q/D_cp
    vol = 0
    for i in range(num_ellipse):
        vol += 1 / math.sqrt(
            np.linalg.det(Q_matrices[i] / D_cp)
        )  ## This should be scaled by the volume of the unit ball, but that's constant so we don't need to use it

    return vol


class Ellipse(ShapeTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.Q = None
        self.center = None

    def fit_shape(self, X):
        delta = 0.0
        num_ellipse = 1
        self.Q, self.center = callCMAESMatrixManyEllipse(X, delta, num_ellipse)

    def score_points(self, X):
        return np.array(
            [np.matmul(np.matmul(x - self.center, self.Q), x - self.center) for x in X]
        )

    def conformalize(self, delta, calibration_data):
        pass

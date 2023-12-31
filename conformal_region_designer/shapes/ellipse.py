import math
import time

import cma
import gurobipy as gp
from matplotlib import axis
import matplotlib.pyplot as plt
import numpy as np

from ..core import ShapeTemplate

PLOT_VALIDATION_TRACES = True
NUM_VALID_TO_PLOT = 100


def ellipse_multiplier(X, Q, center):
    return np.einsum('...i,ij,...j->...', X - center, Q, X - center, optimize='optimal')

def computeCPEllipseMatrixManyEllipse(residuals, Q_matrices, ellipse_centers):
    R_vals = ellipse_multiplier(residuals, Q_matrices[0], ellipse_centers[0])
    r_val_to_ret = np.max(R_vals)
    return r_val_to_ret


def callCMAESMatrixManyEllipse(residuals, delta=1, num_ellipse=1, ellipse_centers=None):
    ## add in cma example
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
        options={"verbose": -9},
        
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

    D_cp = computeCPEllipseMatrixManyEllipse(
        residuals, Q_matrices, ellipse_centers
    )
    return Q_matrices, ellipse_centers, D_cp


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
        residuals, Q_matrices, ellipse_centers
    )
    ## what should the objective function be? sum of volumes? What if the ellipses overlap?

    ## compute area of ellpise given matrices are Q/D_cp
    vol = 0
    for i in range(num_ellipse):
        vol += 1 / math.sqrt(
            np.linalg.det(Q_matrices[i] / D_cp)
        )  ## This should be scaled by the volume of the unit ball, but that's constant so we don't need to use it

    return vol


class EllipsoidTemplate(ShapeTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.Q = None
        self.center = None

    def fit_shape(self, X):
        delta = 0.0
        num_ellipse = 1
        center = X.mean(axis=0)
        self.Q, self.center, self.score_margin = callCMAESMatrixManyEllipse(X, delta, num_ellipse)
        self.center = self.center[0]
        self.Q = self.Q[0]

    def score_points(self, X: np.ndarray):
        score = ellipse_multiplier(X, self.Q/self.score_margin, self.center)
        assert(np.all(score >= 0))
        score = score - 1  # This is because the standard ellipse eq is leq 1.
        return score

    def conformalize(self, delta, calibration_data):
        raise NotImplementedError("Not implemented yet")

    def adjust_shape(self, score_margin):
        self.score_margin = self.score_margin*(score_margin+1.0)
        #assert(self.score_margin >= 0)

    def volume(self):
        return np.pi/np.sqrt(np.linalg.det(self.Q/self.score_margin))

    def plot(self, ax, offset_coords=None, **kwargs):
        pltargs = {"color": "black"}
        pltargs.update(kwargs)
        if self.Q.shape[0] == 3:
            raise NotImplementedError("3d plotting not implemented yet")
        else:
            # Plot the ellipse in 2d
            eigenvalues, eigenvectors = np.linalg.eig(self.Q/self.score_margin)
            theta = np.linspace(0, 2 * np.pi, 1000)
            ellipsis = (1 / np.sqrt(eigenvalues[None, :]) * eigenvectors) @ [
                np.sin(theta),
                np.cos(theta),
            ]
            # print(ellipsis.shape)
            # print(ellipsis[:,0:10])
            if offset_coords is None:
                offset_coords = np.zeros(2)
            ax.plot(
                ellipsis[0, :] + self.center[0] + offset_coords[0],
                ellipsis[1, :] + self.center[1] + offset_coords[1],
                **pltargs
            )


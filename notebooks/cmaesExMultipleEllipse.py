import math
import pickle
import random
import time

import cma
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

# from gurobipy import *


PLOT_VALIDATION_TRACES = True
NUM_VALID_TO_PLOT = 100


def computeCPEllipse(x_vals, y_vals, x_hats, y_hats, a, b, c, delta):
    R_vals = [
        a * (x_vals[i] - x_hats[i]) ** 2
        + 2 * b * (x_vals[i] - x_hats[i]) * (y_vals[i] - y_hats[i])
        + c * (y_vals[i] - y_hats[i]) ** 2
        for i in range(len(x_vals))
    ]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret]


def computeCoverageEllipse(x_vals, y_vals, x_hats, y_hats, a, b, c, D_cp):
    R_vals = [
        a * (x_vals[i] - x_hats[i]) ** 2
        + 2 * b * (x_vals[i] - x_hats[i]) * (y_vals[i] - y_hats[i])
        + c * (y_vals[i] - y_hats[i]) ** 2
        for i in range(len(x_vals))
    ]

    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within) / len(R_vals)
    return coverage_pct


def computeCPCirlce(x_vals, y_vals, x_hats, y_hats, delta):
    R_vals = [
        math.sqrt((x_vals[i] - x_hats[i]) ** 2 + (y_vals[i] - y_hats[i]) ** 2)
        for i in range(len(x_vals))
    ]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret]


def computeCPFixedAlphas(x_vals, y_vals, x_hats, y_hats, alphas, delta):
    R_vals = [
        max(
            [
                alphas[j]
                * math.sqrt(
                    (x_vals[i][j] - x_hats[i][j]) ** 2
                    + (y_vals[i][j] - y_hats[i][j]) ** 2
                )
                for j in range(len(x_vals[i]))
            ]
        )
        for i in range(len(x_vals))
    ]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret]


def computeCoverageRAndAlphas(x_vals, y_vals, x_hats, y_hats, alphas, D_cp):
    R_vals = [
        max(
            [
                alphas[j]
                * math.sqrt(
                    (x_vals[i][j] - x_hats[i][j]) ** 2
                    + (y_vals[i][j] - y_hats[i][j]) ** 2
                )
                for j in range(len(x_vals[i]))
            ]
        )
        for i in range(len(x_vals))
    ]

    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within) / len(R_vals)
    return coverage_pct


def computeCoverageCircle(x_vals, y_vals, x_hats, y_hats, Ds_cp):
    coverage_count = 0

    coverage_count = sum(
        [
            1
            if math.sqrt((x_vals[j] - x_hats[j]) ** 2 + (y_vals[j] - y_hats[j]) ** 2)
            < Ds_cp
            else 0
            for j in range(len(x_vals))
        ]
    )

    coverage_pct = float(coverage_count) / len(x_vals)
    return coverage_pct


def plot_circle(x, y, size, color="-b", label=None):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]

    if label is None:
        plt.plot(xl, yl, color)
    else:
        plt.plot(xl, yl, color, label=label)


def objection_func_cma_es_matrix_ellipse(x, *args):
    # x: [a,b,c]
    a = x[0]
    b = x[1]
    c = x[2]

    if a <= 0 or c <= 0:  # bad cases
        return np.NaN

    if a * c <= b**2:
        return np.NaN

    # args: [x_real,y_real,x_hat,y_hat,delta]

    x_vals = args[0]
    y_vals = args[1]
    x_hats = args[2]
    y_hats = args[3]
    delta = args[4]

    D_cp = computeCPEllipse(x_vals, y_vals, x_hats, y_hats, a, b, c, delta)

    ## Draw ellipse
    # convert to a,c,theta representation
    a_rotated = (
        -1
        / (b**2 / 4 - 4 * a * c)
        * math.sqrt(
            2
            * (b**2 / 4 - 4 * a * c)
            * (-D_cp)
            * ((a + c) + math.sqrt((a - c) ** 2 + b**2 / 4))
        )
    )
    c_rotated = (
        -1
        / (b**2 / 4 - 4 * a * c)
        * math.sqrt(
            2
            * (b**2 / 4 - 4 * a * c)
            * (-D_cp)
            * ((a + c) - math.sqrt((a - c) ** 2 + b**2 / 4))
        )
    )

    area_ellipse = math.pi * a_rotated * c_rotated

    return area_ellipse


def callCMAES(x_real, y_real, x_hat, y_hat, delta):
    ## add in cma example

    args_cma = [x_real, y_real, x_hat, y_hat, delta]

    x_0 = [1, 0, 1]  # a,b,c
    sigma0 = 0.25  # [0.25,1,0.25] # the std of the search space. Should be about 1/4 of size of search space. This is a complete guess

    start_time = time.time()
    x, es = cma.fmin2(objection_func_cma_es_matrix_ellipse, x_0, sigma0, args=args_cma)
    end_time = time.time()

    print("cmaes soln: " + str(x))
    print("Soln time: " + str(end_time - start_time))

    print("a: " + str(x[0]))
    print("b: " + str(x[1]))
    print("c: " + str(x[2]))

    return x[0], x[1], x[2]


## rewrite cmaes and ellipse code for arbitrary dimensions?


def computeCPEllipseMatrix(gt_vals, pred_vals, Q, delta):
    R_vals = [
        np.matmul(np.matmul(gt_vals[i] - pred_vals[i], Q), gt_vals[i] - pred_vals[i])
        for i in range(gt_vals.shape[0])
    ]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret].item()


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

    residuals = np.array(residuals)

    # gt_vals = np.array(gt_vals)
    # pred_vals = np.array(pred_vals)
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

    # x_0 = np.identity(output_dim)
    # x_0 = x_0.reshape(x_0.size) # input to cma needs to be an array
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
    # gt_vals = args[0]
    # pred_vals = args[1]
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

    # D_cp = computeCPEllipseMatrix(gt_vals,pred_vals,Q,delta)
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


def sample_circle_unif(center, radius, num_points, x_stretch=1, y_stretch=1):
    samples = []
    for _ in range(num_points):
        # sample angle
        # sample distance
        angle = np.random.uniform(0, 2 * math.pi)
        rad = np.random.uniform(0, radius)
        point = [
            center[0] + rad * x_stretch * math.cos(angle),
            center[1] + rad * y_stretch * math.sin(angle),
        ]
        samples.append(point)
    return samples


def plot_ellipse(center, Q, D_cp, color, label):
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    theta = np.linspace(0, 2 * np.pi, 1000)
    ellipsis = (1 / np.sqrt(eigenvalues[None, :] / D_cp) * eigenvectors) @ [
        np.sin(theta),
        np.cos(theta),
    ]
    # print(ellipsis.shape)
    # print(ellipsis[:,0:10])
    plt.plot(ellipsis[0, :] + center[0], ellipsis[1, :] + center[1], color, label=label)


def main_fake_initial_example():
    ## generate data (just draw points uniformly from two separate circles, then call the optimization)

    circle_1_center = [3, 0]
    circle_2_center = [-3, 0]

    circle_1_radius = 1
    circle_2_radius = 2

    ellipse_1_center = [0, -10]
    ellipse_2_center = [-3, 0]
    ellipse_centers = [ellipse_1_center, ellipse_2_center]

    num_samples = 100
    samples_circle_1 = sample_circle_unif(
        circle_1_center, circle_1_radius, num_samples, x_stretch=2
    )
    samples_circle_2 = sample_circle_unif(circle_2_center, circle_2_radius, num_samples)

    num_ellipse = 2
    delta = 0.05

    samples = samples_circle_1
    samples.extend(samples_circle_2)

    fake_GT_vales = [[0, 0] for _ in range(len(samples))]

    samples = np.array(samples)
    fake_GT_vales = np.array(fake_GT_vales)

    Q_matrices, ellipse_centers = callCMAESMatrixManyEllipse(
        samples, delta, num_ellipse, ellipse_centers=ellipse_centers
    )

    Q_matrices = np.array(Q_matrices)
    ellipse_centers = np.array(ellipse_centers)

    for i in range(num_ellipse):
        print(ellipse_centers[i])
        print(Q_matrices[i])

    D_cp = computeCPEllipseMatrixManyEllipse(
        samples, Q_matrices, ellipse_centers, delta
    )
    print("D_cp: " + str(D_cp))

    plt.scatter([s[0] for s in samples], [s[1] for s in samples])
    plot_ellipse(ellipse_centers[0], Q_matrices[0], D_cp, "green", "ellipse 1")
    plot_ellipse(ellipse_centers[1], Q_matrices[1], D_cp, "red", "ellipse 2")
    plt.legend()

    plt.savefig("multiEllipsePlot.png")


if __name__ == "__main__":
    main_fake_initial_example()

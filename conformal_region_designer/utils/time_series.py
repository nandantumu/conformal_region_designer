import math

import gurobipy as gp
import numpy as np
from gurobipy import GRB

# from gurobipy import *



def compute_optimized_alphas(R_vals, delta, connection_params=None):
    """
    This function solves an optimization problem to obtain an optimal weighting 
    of the non-conformity scores for the time-weighted case. This method is due 
    to the paper: Conformal Prediction Regions for "Time Series using Linear 
    Complementarity Programming", by Cleaveland et. al. (2023)
    https://arxiv.org/abs/2304.01075

    R_vals: n x T dim array. n data points, each with T non-conformity scores
    delta: total desired coverage
    """
    M = 100_000
    n = len(R_vals)
    T = len(R_vals[0])
    delta = 1 - delta

    ## R_vals: n x T dim array. n data points, each with T non-conformity scores

    with gp.Env(params=connection_params) as env, gp.Model(env=env) as m:
        m.setParam("TimeLimit", 5*60.0)
        try:
            ## declare variables we have
            # q, continuous, positive
            # alpha_t, continuous, positive, one for each of T time steps in time series
            # e^+_i, e^-_i, continuous, positive, one for each of n time series
            # b_t^i, binary, one for each of T time steps for each of n time series (T x n total)
            # R^i, continuous, positive, one for each of n time series

            es_plus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="es_plus")
            es_minus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="es_minus")
            b = m.addVars(n, T, vtype=GRB.BINARY, name="b")
            Rs = m.addVars(n, vtype=GRB.CONTINUOUS, name="Rs")

            us_plus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_plus")
            us_minus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_minus")
            v = m.addVars(n, lb=-float("inf"), vtype=GRB.CONTINUOUS, name="v")

            q = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="q")
            alphas = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="alphas")

            ## create objective
            obj = gp.LinExpr(q)
            m.setObjective(obj, GRB.MINIMIZE)

            #### KKT constraints

            ## gradients are 0 (stationary condition)
            q_gradient_constraint = gp.LinExpr()
            for i in range(n):
                m.addConstr((1 - delta) - us_plus[i] + v[i] == 0)
                m.addConstr(delta - us_minus[i] - v[i] == 0)
                q_gradient_constraint += v[i]
            m.addConstr(q_gradient_constraint == 0)

            ## complementary slackness
            for i in range(n):
                m.addConstr(us_plus[i] * es_plus[i] == 0)
                m.addConstr(us_minus[i] * es_minus[i] == 0)

            # ## primal feasibility
            for i in range(n):
                m.addConstr(es_plus[i] + q - es_minus[i] - Rs[i] == 0)
                m.addConstr(es_plus[i] >= 0)
                m.addConstr(es_minus[i] >= 0)

            # ## dual feasibility
            for i in range(n):
                m.addConstr(us_plus[i] >= 0)
                m.addConstr(us_minus[i] >= 0)

            for i in range(n):
                for t in range(T):
                    m.addConstr(Rs[i] >= alphas[t] * R_vals[i][t])
                    m.addConstr(Rs[i] <= alphas[t] * R_vals[i][t] + (1 - b[(i, t)]) * M)

            for i in range(n):
                b_constraint = gp.LinExpr()
                for t in range(T):
                    b_constraint += b[(i, t)]
                m.addConstr(b_constraint == 1)

            m_constraint = gp.LinExpr()
            for t in range(T):
                m_constraint += alphas[t]
                m.addConstr(alphas[t] >= 0)
            m.addConstr(m_constraint == 1)

            m.optimize()

            alphas = []
            for v in m.getVars():
                if v.varName == "q" or "alphas" in v.varName:
                    if "alphas" in v.varName:
                        alphas.append(v.x)
            
            alphas = np.array(alphas)
            print("Obj: " + str(m.objVal))

        except gp.GurobiError as e:
            print("Error from gurobi: " + str(e))
            alphas = np.ones(n)/n

    return alphas

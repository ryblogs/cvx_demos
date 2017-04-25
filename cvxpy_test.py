#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:28:04 2017

@author: soolr
"""

import cvxpy as cvx
import numpy as np

# Problem data.
m = 50
n = 10
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

def cvxit(A,b):
    # Construct the problem.
    x = cvx.Variable(A.shape[1])
    objective = cvx.Minimize(cvx.sum_squares(A*x - b))
    constraints = [0 <= x, x <= 1]
    prob = cvx.Problem(objective, constraints)
    
    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    #print(x.value)
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    #print(constraints[0].dual_value)
    return np.array(x.value).flatten()

X = cvxit(A,b)
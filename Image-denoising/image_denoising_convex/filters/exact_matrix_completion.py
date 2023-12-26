# Exact Matrix Completion
# complete_matrix() should work with any matrix, symmetric or not.
# complete_psd_symmetric() will work with psd symmetric matrices only, but is faster.

import numpy as np
import cvxpy as cp
import random

def complete_psd_symmetric(M, omega):
    """
    If M is already symmetric PSD, can safely use this
    to recover the matrix.
    """
    X = cp.Variable(M.shape, PSD=True)

    constraints = [X == X.T] # symmetry constraint
    for i, j in omega:
        constraints += [X[i, j] == M[i, j]] # equality constraint for sampled entries

    # Minimize surrogate of nuclear norm: trace
    problem = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    problem.solve()

    return X.value

def complete_matrix(M, omega):
    """
    If M is not guaranteed to by symmetric PSD, use this function instead.
    """

    """
    create a decision variable which has shape 2n X 2n essentially,
    | W1 X  |
    | X* W2 |
    we want to minimize Tr(W1) + Tr(W2), but only really care about X as this will be our original reconstructed matrix. 
    The reason for introducing W1, W2, and X* is because we want to minimize the trace of a symmetric PSD matrix.
    """
    
    X = cp.Variable([np.sum(M.shape), np.sum(M.shape)], PSD=True) # create the 2n X 2n matrix

    constraints = [X == X.T] # symmetry constraint
    for i, j in omega:
        constraints += [X[i, j + M.shape[0]] == M[i, j]] # equality constraint for sampled entries

    # Minimize surrogate of nuclear norm: trace(X)
    problem = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    problem.solve()

    # return top right corner of matrix
    return X.value[:M.shape[0], M.shape[0]:]

def mask_out_matrix(X, entries):
    mask = np.zeros(X.shape)
    omega = random.sample([(i, j) for i in range(X.shape[0]) for j in range(X.shape[1])], entries)
    for (i, j) in omega:
        mask[i, j] = 1
    return X.copy() * mask, omega

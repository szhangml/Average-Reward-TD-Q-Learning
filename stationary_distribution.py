#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find the stationary distribution of a Markov Reward Process (MRP)
"""

import numpy as np
from numpy import linalg as LA
from randomMRP import GenerateRandomMRP


def stationary_distribution(trans_prob):
    """
        Find the stationary distribution for an MRP

        Inputs:
                trans_prob: transition probability matrix.

        Output:
                pi: stationary distribution.
    """

    # find the eigenvalues and left eigenvectors of P
    eigvalues, eigvectors = LA.eig(trans_prob.T)

    # find the stationary distribution
    for idx, eigval in enumerate(eigvalues):
        if abs(1.0 - eigval.real) < 1e-8:
            pi = eigvectors[:, idx].real
            pi = pi / sum(pi)
            assert np.allclose(np.dot(pi, trans_prob), pi), "Cannot find the stationary distribution."
            return pi

    raise Exception('Cannot find the stationary distribution')


if __name__ == "__main__":
    num_states = 10
    mrp = GenerateRandomMRP(num_states)
    pi = stationary_distribution(mrp.trans_prob)
    print(pi)
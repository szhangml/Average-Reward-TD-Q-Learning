#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate an n-by-d feature matrix.
"""

import numpy as np
from numpy import linalg as LA
from randomMRP import GenerateRandomMRP
from stationary_distribution import stationary_distribution
from gain_and_bias import stationary_reward, basic_bias


def generate_feature_matrix(num_states, d, bias):
    """
        Inputs:
                (1) num_states: number of states
                (2) d: number of features
                (3) bias: basic differential value function
        Output:
                A feature matrix that satisfies
                (1) Full column rank: rank(Phi) = d
                (2) Normalized rows: || Phi(i,:) ||_2 <= 1, for every  i
                (3) e \in Col(\Phi)
                (4) bias \in Col(\Phi)
    """

    for trial in range(100):
        feature_matrix = np.random.binomial(1, 0.5, (num_states, d-2))
        e = np.ones(num_states)
        feature_matrix = np.column_stack((feature_matrix, e, bias))

        if LA.matrix_rank(feature_matrix) == d and np.min(LA.norm(feature_matrix, axis=1)) > 0:
            return feature_matrix / np.max(LA.norm(feature_matrix, axis=1))

    raise Exception('Cannot generate a proper feature matrix. Please try again!')


if __name__ == "__main__":
    mrp = GenerateRandomMRP(num_states=20)
    pi = stationary_distribution(mrp.trans_prob)
    gain = stationary_reward(mrp.rewards, pi)
    bias = basic_bias(mrp.rewards, mrp.trans_prob, gain, pi)
    Phi = generate_feature_matrix(mrp.num_states, 5, bias)
    print(Phi)
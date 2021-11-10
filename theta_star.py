#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
find theta_star.
"""

import numpy as np
from numpy import linalg as LA
from randomMRP import GenerateRandomMRP
from stationary_distribution import stationary_distribution
from gain_and_bias import stationary_reward, basic_bias
from feature_matrix import generate_feature_matrix


def find_theta_star(Phi, bias):
    """
        Inputs:
                (1) Phi: feature matrix.
                (2) bias: basic differential value function
        Output:
                theta_star
    """

    return LA.lstsq(Phi, bias, rcond=None)[0]


if __name__ == '__main__':
    mrp = GenerateRandomMRP(num_states=100)
    pi = stationary_distribution(mrp.trans_prob)
    gain = stationary_reward(mrp.rewards, pi)
    bias = basic_bias(mrp.rewards, mrp.trans_prob, gain, pi)
    Phi = generate_feature_matrix(mrp.num_states, 10, bias)
    theta_star = find_theta_star(Phi, bias)

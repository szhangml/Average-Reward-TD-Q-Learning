#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compute the gain and bias of a Markov reward process (MRP).
Gain is the average reward per time period.
Bias (differential value function) is the expected total difference between the reward and the gain.
"""

import numpy as np
from randomMRP import GenerateRandomMRP
from stationary_distribution import stationary_distribution
from numpy import linalg as LA


def stationary_reward(rewards, pi):
    """
        Inputs:
                (1) rewards: vector of rewards.
                (2) pi: stationary distribution.

        Output:
                gain: average reward of the MRP.
    """
    return np.dot(rewards, pi)


def basic_bias(rewards, trans_prob, gain, pi):
    """
        Inputs:
                (1) rewards: vector of rewards.
                (2) trans_prob: transition probability matrix.
                (3) gain: average reward.
                (4) pi: stationary distribution.

        Output:
                the bias that satisfies np.dot(bias, pi) = 0.
    """
    n = rewards.shape[0]
    b = np.append(rewards-gain*np.ones(n), 0)
    A = np.row_stack((np.identity(n)-trans_prob, pi))
    sol = LA.lstsq(A, b, rcond=None)
    return sol[0]


if __name__ == '__main__':
    mrp = GenerateRandomMRP(num_states=10)
    pi = stationary_distribution(mrp.trans_prob)
    gain = stationary_reward(mrp.rewards, pi)
    bias = basic_bias(mrp.rewards, mrp.trans_prob, gain, pi)
    print(gain, bias)
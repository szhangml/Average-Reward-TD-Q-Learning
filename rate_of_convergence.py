#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diminishing step-size rate of convergence
"""

import numpy as np
from numpy import linalg as LA
from randomMRP import GenerateRandomMRP
from feature_matrix import generate_feature_matrix
from stationary_distribution import stationary_distribution
from gain_and_bias import stationary_reward, basic_bias
from theta_star import find_theta_star


def linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes):
    """
        Inputs:
                (1) mrp: markov reward process
                (2) Phi: feature matrix
                (3) theta_star: TD fixed point
                (4) theta_e: solution of Phi theta = e
                (5) Lambda: algorithm lambda parameter
                (4) T: number of iterations
                (5) c_alpha: algorithm step size parameter
                (6) step_sizes: step-size sequence

        Output:
                se_hist: array of squared error that measures the progress of the algorithm

    """

    d = Phi.shape[1]

    # Initialization
    bar_r = 0.0
    theta = np.zeros(d)
    z = np.zeros(d)
    se_hist = np.zeros(T)
    mrp.reset_initial_state()

    for t in range(T):
        # Observe data
        current_state = mrp.current_state
        r, next_state = mrp.step()

        # Get TD error
        delta = r - bar_r + np.dot(Phi[next_state, :], theta) - np.dot(Phi[current_state, :], theta)

        # Update eligibility trace
        z = Lambda * z + Phi[current_state, :]

        # Update average-reward estimate
        bar_r = bar_r + c_alpha * step_sizes[t] * (r - bar_r)


        # Update parameter vector
        theta = theta + step_sizes[t] * delta * z
        projected_theta = theta - (np.dot(theta, theta_e) / np.dot(theta_e, theta_e)) * theta_e

        # squared_error
        se = (bar_r - gain)**2 + LA.norm(projected_theta - theta_star)**2
        se_hist[t] = se

    return se_hist


if __name__ == "__main__":
    mrp = GenerateRandomMRP(num_states=100)
    pi = stationary_distribution(mrp.trans_prob)
    gain = stationary_reward(mrp.rewards, pi)
    bias = basic_bias(mrp.rewards, mrp.trans_prob, gain, pi)
    d = 20
    Phi = generate_feature_matrix(mrp.num_states, d, bias)
    theta_star = find_theta_star(Phi, bias)
    theta_e = LA.lstsq(Phi, np.ones(mrp.num_states), rcond=None)[0]
    T = 1000000
    c_alpha = 1
    step_sizes = 150 / (np.arange(T) + 1000)
    num_exp = 100

    for Lambda in [0, 0.2, 0.4, 0.8]:
        se_exp = np.zeros((num_exp, T))
        for exp in range(num_exp):
            print(f"Lambda = {Lambda}, exp = {exp}")
            se_hist = linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes)
            se_exp[exp, :] = se_hist
        mse = np.mean(se_exp, axis=0)



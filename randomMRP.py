#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a random Markov reward process (MRP).
"""

import numpy as np


def GenerateRandomMRP(num_states):
    """
        Inputs:
                (1) num_states: number of states.

        Output:
                An MRP with
                (1) State space S = [0,1,...,num_states-1].
                (2) Immediate reward r(s) for each state. Each reward r(s) is drawn from Uniform(0,1).
                (3) Transition probability P(s,s') = P(next_state=s'| current_state=s).
                    For any state, the probabilities of each successor state s'
                    are chosen as random partitions of the unit interval.
    """

    rewards = generate_rewards(num_states)
    trans_prob = generate_transition_probability(num_states)

    return MRP(num_states, rewards, trans_prob)


def generate_rewards(num_states):
    """
        Generate the reward for each state.

        Inputs:
                num_states: number of states

        Output:
                A reward vector.
    """
    return np.random.uniform(0, 1, num_states)


def generate_transition_probability(num_states):
    """
        Generate the transition probabilities for each state by random partitions of the unit interval.

        Inputs:
                num_states: number of states

        Output:
                trans_prob: transition probability matrix.
    """
    trans_prob = np.zeros((num_states, num_states))

    for s in range(num_states):
        trans_prob[s, :] = next_state_prob(num_states)

    return trans_prob


def next_state_prob(num_states):
    """
        Generate the next-state probability by random partitions of the unit interval.

        Input:
                num_states: number of next states

        Output:
                prob: a vector of next-state probabilities.
    """

    rvs = np.random.uniform(0, 1, num_states - 1)
    rvs.sort()
    prob = np.zeros(num_states)

    for s in range(num_states):
        if s == 0:
            prob[s] = rvs[0] - 0.0
        elif s == num_states - 1:
            prob[s] = 1.0 - rvs[-1]
        else:
            prob[s] = rvs[s] - rvs[s - 1]

    return prob


class MRP(object):

    def __init__(self, num_states, rewards, trans_prob):
        """
                Inputs:
                        (1) num_states: number of states
                        (2) rewards: reward vector
                        (3) trans_prob: transition probability matrix.

                Output:
                        An MRP with
                        (1) State space S = {0,1,...,num_states-1}.
                        (2) Reward r(s) for each state s.
                        (3) Transition probability p(s,s') = P(next_state=s'| current_state=s).
        """

        self.num_states = num_states
        self.state_space = list(range(num_states))
        self.rewards = rewards
        self.trans_prob = trans_prob
        self.current_state = None
        self.reset_initial_state()

    def next_state(self, s):
        """
            Return a sample next state when it is in state s.
        """
        return np.random.choice(self.state_space, size=None, p=self.trans_prob[s, :])

    def reward(self, s):
        """
            Return the immediate reward when it is in state s.
        """
        return self.rewards[s]

    def step(self):
        """
            Return the immediate reward and the random next state.
        """
        r = self.reward(self.current_state)
        next_state = self.next_state(self.current_state)

        self.current_state = next_state  # update the current state

        return r, next_state

    def reset_initial_state(self):
        """
            randomly reset the initial state.
        """
        self.current_state = np.random.choice(self.state_space, size=None, p=None)


if __name__ == "__main__":
    mrp = GenerateRandomMRP(num_states=5)
    print(mrp.__dict__)
    print(mrp.step())

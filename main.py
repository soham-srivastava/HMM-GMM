import pandas as pd
import numpy as np

class HMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        self.pi = np.ones(n_states) / n_states  # Initial state probabilities
        self.A = np.ones((n_states, n_states)) / n_states  # State transition probabilities
        self.B = np.ones((n_states, n_observations)) / n_observations  # Emission probabilities (it is the probability to observe a certain observation given a state)

    def fit(self, observations):
        # Implement the Baum-Welch algorithm to estimate the parameters
        pass
    def predict(self, observations):
        # Implement the Viterbi algorithm to find the most likely sequence of hidden states
        pass
    # what about forward backward algorithum ?
    def forward_backward(self, observations):
        # Implement the Forward-Backward algorithm to compute the posterior probabilities of the hidden states
        pass

import gym
from gym import spaces
import numpy as np
from module import *

class BEROptimizationEnv(gym.Env):
    def __init__(self, input_length=1024, snr_values=None):
        super(BEROptimizationEnv, self).__init__()

        if snr_values is None:
            self.snr_values = [0.2 * i for i in range(11)]
        else:
            self.snr_values = snr_values
        
        self.input_length = input_length
        self.rows_choices = [i for i in range(1, input_length + 1) if input_length % i == 0]
        self.cols_choices = [input_length // i for i in self.rows_choices]
        self.action_space = spaces.Discrete(len(self.rows_choices))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.current_state = np.array([0.5])
        self.previous_ber = 1.0
        self.best_ber = float('inf')
        self.best_rows = None
        self.best_cols = None
        self.visited_configs = set()

    def step(self, action_idx):
        rows = self.rows_choices[action_idx]
        cols = self.cols_choices[action_idx]
        input_data = np.random.randint(0, 2, self.input_length)

        ber_values = []
        for snr in self.snr_values:
            ber = simulate_hybrid_interleaver(input_data, snr, soft_decision_decoder)
            ber_values.append(ber)

        average_ber = np.mean(ber_values)
        reward = (self.previous_ber - average_ber) * 100

        if (rows, cols) in self.visited_configs:
            reward += 0.5

        self.previous_ber = average_ber
        self.visited_configs.add((rows, cols))
        self.current_state = np.array([average_ber])

        if average_ber < self.best_ber:
            self.best_ber = average_ber
            self.best_rows = rows
            self.best_cols = cols

        done = average_ber < 0.01
        return self.current_state, reward, done, {}

    def reset(self):
        self.current_state = np.array([1.0])
        self.previous_ber = 1.0
        self.best_ber = float('inf')
        self.best_rows = None
        self.best_cols = None
        self.visited_configs = set()
        return self.current_state

    def get_best_configuration(self):
        return self.best_rows, self.best_cols, self.best_ber

    def close(self):
        pass

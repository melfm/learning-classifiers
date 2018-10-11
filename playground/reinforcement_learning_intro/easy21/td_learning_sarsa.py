"""TD Learning.
"""
import numpy as np
from tqdm import tqdm

import easy21_environment as easyEnv


class SarsaAgent:

    def __init__(self, environment, num_episodes=1000, n0=100,
                 td_lambda=0):
        self.env = environment
        self.num_episodes = num_episodes
        self.td_lambda = td_lambda
        # This is a constant-hyperparameter.
        self.N0 = float(n0)

        self.Q = np.zeros((self.env.dealer_value_count,
                           self.env.player_value_count,
                           self.env.action_count))

        # N(s) is book-keeping for the number of times state
        # s has been visited. N(s,a) is the number of times
        # action a has been selected from state s.
        self.N = np.zeros((self.env.dealer_value_count,
                           self.env.player_value_count,
                           self.env.action_count))

        # Initialise the value function to zero.
        self.V = np.zeros((self.env.dealer_value_count,
                           self.env.player_value_count))

        self.eligibility = np.zeros((self.env.dealer_value_count,
                                     self.env.player_value_count,
                                     self.env.action_count))

        self.player_wins = 0
        self.episodes = 0

    def get_epsilon(self, N):
        return self.N0 / (self.N0 + N)

    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy exploration strategy.

        Args:
            state: State object representing the status of the game.

        Retur1ns:
            action: Chosen action based on Epsilon-greedy.
        """
        dealer = state.dealer_sum - 1
        player = state.player_sum - 1
        epsilon = self.get_epsilon(sum(self.N[dealer, player, :]))
        if np.random.rand() < (epsilon):

            action = np.argmax(self.Q[dealer, player, :])
        else:
            action = np.random.choice(easyEnv.ACTIONS)

        return action

    def train(self):
        """TD-Sarsa training.
        """

        for episode in tqdm(range(self.num_episodes)):

            # Initialize the state
            state = self.env.init_state()
            action = self.epsilon_greedy_policy(state)
            next_action = action

            while not state.terminal:

                # Execute the action
                next_state, reward = self.env.step(state, action)
                # State-action index
                idx = state.dealer_sum - 1, state.player_sum - 1, action

                if not next_state.terminal:
                    next_action = self.epsilon_greedy_policy(next_state)
                    next_idx = next_state.dealer_sum - 1, \
                        next_state.player_sum - 1, next_action

                    td_error = reward + self.Q[next_idx] - self.Q[idx]
                else:
                    td_error = reward - self.Q[idx]

                self.N[idx] += 1
                self.eligibility[idx] += 1
                # Step-size
                alpha = 1.0 / self.N[idx]

                self.eligibility[idx] *= self.td_lambda

                # Sarsa update
                self.Q[idx] += alpha * (td_error) * self.eligibility[idx]

                state = next_state
                action = next_action

            if reward == 1:
                self.player_wins += 1

        for d in range(self.env.dealer_value_count):
            for p in range(self.env.player_value_count):
                self.V[d, p] = max(self.Q[d, p, :])

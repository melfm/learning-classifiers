"""Monte-Carlo Control.
"""
import numpy as np
from tqdm import tqdm

import easy21_environment as easyEnv


class MonteCarloAgent:

    def __init__(self, environment, num_episodes=10000, n0=100):
        self.env = environment
        self.num_episodes = num_episodes
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

        self.V = np.zeros((self.env.dealer_value_count,
                           self.env.player_value_count))

        self.player_wins = 0
        self.episodes = 0

    def get_epsilon(self, N):
        return self.N0 / (self.N0 + N)

    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy exploration strategy.

        Args:
            state: State object representing the status of the game.

        Returns:
            action: Chosen action based on Epsilon-greedy.
        """
        dealer = state.dealer_sum
        player = state.player_sum
        epsilon = self.get_epsilon(sum(self.N[dealer, player, :]))
        if np.random.rand() < (epsilon):

            action = np.argmax(self.Q[dealer, player, :])
        else:
            action = np.random.choice(easyEnv.ACTIONS)

        return action

    def train(self):
        """Monte-Carlo training.

        Update the state-action function Q, after seeing the
        whole trajectory.
        """

        for episode in tqdm(range(self.num_episodes)):
            player_trajectories = []

            # Initialize the state
            state = self.env.init_state()

            while not state.terminal:
                action = self.epsilon_greedy_policy(state)
                player_trajectories.append((state, action))

                # Book-keeping the visits
                idx = state.dealer_sum, state.player_sum, action
                self.N[idx] += 1

                # Execute the action
                next_state, reward = self.env.step(state, action)
                state = next_state

            if reward == 1:
                self.player_wins += 1

            # Update action-value function Q
            for state, action in player_trajectories:
                idx = state.dealer_sum, state.player_sum, action
                alpha = 1.0 / self.N[idx]
                self.Q[idx] += alpha * (reward - self.Q[idx])

        for d in range(self.env.dealer_value_count):
            for p in range(self.env.player_value_count):
                self.V[d, p] = max(self.Q[d, p, :])

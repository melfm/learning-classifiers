"""Monte-Carlo Control.
"""
import numpy as np
from matplotlib import cm

import easy21_environment as easyEnv
import utils as utils


class MonteCarloAgent:

    def __init__(self, environment, num_episodes=1000, n0=100):
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
        return self.N0 / self.N0 + N

    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy exploration strategy.

        Args:
            Q:
            N:
            state:
        """
        dealer = state.dealer_sum
        player = state.player_sum
        # pdb.set_trace()
        epsilon = self.get_epsilon(np.sum(self.N[dealer - 1,
                                                 player - 1,
                                                 :]))
        if np.random.rand() < (1 - epsilon):
            print('Epsilon greedy, picking best action')
            action = np.argmax(self.Q[dealer - 1,
                                      player - 1,
                                      :])
        else:
            print('Epsilon greedy, exploring')
            action = np.random.choice(easyEnv.ACTIONS)

        return action

    def train(self):
        """Monte-Carlo training.

        Update the state-action function Q, after seeing the
        whole trajectory.
        """

        for episode in range(1, self.num_episodes + 1):
            player_trajectories = []

            # Initialize the state
            state = self.env.init_state()

            while not state.terminal:
                action = self.epsilon_greedy_policy(state)
                player_trajectories.append((state, action))

                # Book-keeping the visits
                idx = state.dealer_sum - 1, state.player_sum - 1, action
                self.N[idx] += 1

                # Execute the action
                next_state, reward = self.env.step(state, action)
                state = next_state

            if reward == 1:
                self.player_wins += 1

            # Update action-value function Q
            for state, action in player_trajectories:
                idx = state.dealer_sum - 1, state.player_sum - 1, action
                alpha = 1.0 / self.N[idx]
                self.Q[idx] += alpha * (reward - self.Q[idx])

        print('Win rate ', (float(self.player_wins)/self.num_episodes) * 100)

        for d in range(self.env.dealer_value_count):
            for p in range(self.env.player_value_count):
                self.V[d, p] = max(self.Q[d, p, :])

    def plot_frame(self, ax):
        def get_stat_val(x, y):
            return self.V[x, y]

        X = np.arange(0, self.env.dealer_value_count, 1)
        Y = np.arange(0, self.env.player_value_count, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_stat_val(X, Y)
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False)
        return surf

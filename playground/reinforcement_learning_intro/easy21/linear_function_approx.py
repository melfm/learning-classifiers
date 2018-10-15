"""TD Learning.
"""
import numpy as np
from functools import reduce
from tqdm import tqdm

import easy21_environment as easyEnv


class LFASarsaAgent:
    """Linear Function Approximation Sarsa Agent.
    """

    def __init__(self, environment, num_episodes=1000, td_lambda=0):
        self.env = environment
        self.num_episodes = num_episodes
        self.td_lambda = td_lambda
        self.alpha = 0.01
        self.epsilon = 0.05

        self.cuboid_intervals = {
            'dealer': ((1, 4), (4, 7), (7, 10)),
            'player': ((1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)),
            'action': ((0), (1))
        }

        # Dimension (3, 6, 2)
        self.feature_dim = (3, 6, 2)

        self._reset()

        self.player_wins = 0

    def _reset(self):

        # Initialise the value function to zero.
        self.V = np.zeros((self.env.dealer_value_count,
                           self.env.player_value_count))

        # Smaller weights help with stabilization
        self.W = np.random.rand(reduce((lambda x, y: x * y),
                                       (self.feature_dim)), 1) * 0.001

        # These eligbilities are on the feature space now, so same
        # size as the weights.
        self.eligibility = np.zeros_like(self.W)

    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy exploration strategy.

        Args:
            state: State object representing the status of the game.

        Returns:
            action: Chosen action based on Epsilon-greedy.
            qhat: Estimated features.
        """
        if np.random.rand() < (self.epsilon):
            qhat, action = max(((np.dot(self.make_features(state, a),
                                        self.W), a)
                                for a in easyEnv.ACTIONS), key=lambda x: x[0])
        else:
            action = np.random.choice(easyEnv.ACTIONS)
            # This is the same as sum and multiplication (dimensions need to be
            # correct for that though.)
            qhat = np.dot(self.make_features(state, action), self.W)

        return action, qhat

    def make_features(self, state, action):
        """Each binary feature has a value of 1 iff (s, a) lies within
         the cuboid of state-space corresponding to that feature,
         and the action corresponding to that feature.
        """
        if state.terminal:
            return 0

        dealer_sum, player_sum = state.dealer_sum, state.player_sum

        state_features = np.zeros((self.feature_dim[0], self.feature_dim[1]),
                                  dtype=int)
        for di_idx, di in enumerate(self.cuboid_intervals['dealer']):
            for pi_idx, pi in enumerate(self.cuboid_intervals['player']):
                if di[0] <= dealer_sum <= di[1] and\
                        pi[0] <= player_sum <= pi[1]:
                    state_features[di_idx][pi_idx] = 1

        """
        # More compact version of the above loop.
        state_features = np.array([
                (di[0] <= dealer_sum <= di[1]) and \
                    (pi[0] <= playe_sum <= pi[1])
                for di in self.cuboid_intervals['dealer']
                for pi in self.cuboid_intervals['player']
            ]).astype(int).reshape(self.feature_dim[:2])
        """

        phi = np.zeros((self.feature_dim), dtype=int)
        for action_idx, act in enumerate(self.cuboid_intervals['action']):
            if action == act:
                phi[:, :, action_idx] = state_features
        # Flatten the features
        phi = phi.reshape(1, -1)

        return phi.astype(int)

    def combine_final_features(self):
        dealer_count = self.env.dealer_value_count
        player_count = self.env.player_value_count
        action_count = self.env.action_count

        linear_comb_features = np.zeros((dealer_count,
                                         player_count,
                                         action_count))

        for dealer in range(0, dealer_count):
            for player in range(0, player_count):
                for action in range(0, action_count):
                    state = easyEnv.State(dealer, player)
                    phi = self.make_features(state, action)
                    # Represent action-value function by a linear
                    # combination of features
                    linear_comb_features[dealer, player, action] = \
                        np.dot(phi, self.W)
        return linear_comb_features

    def train(self, mc_agent_q, run_single_lambda=False):
        """TD-Sarsa training.

        Args:
            mc_agent_q: True values Q ∗ (s, a), computed by Monte-Carlo.
            run_single_lambda: Flag to only run one iteration for the
                assigned lambda (upon agent instantiation).

        Returns:
            mse_per_lambdas: Mean squared error per episode. The error
                is compared against mc_agent_q state-action values.
        """

        if run_single_lambda:
            td_lambdas = np.arange(self.td_lambda, self.td_lambda+1, 1)
        else:

            td_lambdas = np.arange(0, 1.10, 0.1)

        mse_per_lambdas = np.zeros((len(td_lambdas), self.num_episodes))
        end_of_episode_mse = np.zeros(len(td_lambdas))

        for li, lam in enumerate(td_lambdas):
            self._reset()

            for episode in tqdm(range(self.num_episodes)):
                self.eligibility = np.zeros_like(self.W)

                # Initialize the state
                state = self.env.init_state()
                action, qhat = self.epsilon_greedy_policy(state)
                next_action = action

                while not state.terminal:

                    # Execute the action
                    next_state, reward = self.env.step(state, action)

                    if not next_state.terminal:
                        next_action, next_qhat = \
                            self.epsilon_greedy_policy(next_state)
                        td_error = reward + next_qhat - qhat

                    else:
                        td_error = reward - qhat

                    # Update = step-size × prediction error × feature value
                    # E_t = lambda * E_{t-1} + x(S_t)
                    self.eligibility = lam * self.eligibility + \
                        self.make_features(state, action).reshape(-1, 1)
                    dw = self.alpha * td_error * self.eligibility
                    # Adjust the weights
                    self.W += dw

                    state = next_state
                    action = next_action

                if reward == 1:
                    self.player_wins += 1

                # Combine all the features we just learned.
                sarsa_q = self.combine_final_features()
                mse_term = np.sum((sarsa_q - mc_agent_q)
                                  ** 2) / np.size(sarsa_q)

                mse_per_lambdas[li, episode] = mse_term

                if episode % 1000 == 0 or episode+1 == self.num_episodes:
                    print("Lambda=%.1f Episode %06d, MSE %5.3f" % (
                        lam, episode, mse_term))

            end_of_episode_mse[li] = mse_term
            for d in range(self.env.dealer_value_count):
                for p in range(self.env.player_value_count):
                    self.V[d, p] = max(sarsa_q[d, p, :])

        return mse_per_lambdas, end_of_episode_mse

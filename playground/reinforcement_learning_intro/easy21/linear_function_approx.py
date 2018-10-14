"""TD Learning.
"""
import numpy as np
from functools import reduce
from tqdm import tqdm

import easy21_environment as easyEnv
import pdb

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

        self.W = np.random.rand(reduce((lambda x, y:x * y),
                                       (self.feature_dim)), 1)

        self.eligibility = np.zeros_like(self.W)

        self.linear_comb_features = self.linearly_combine_features()

    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy exploration strategy.

        Args:
            state: State object representing the status of the game.

        Retur1ns:
            action: Chosen action based on Epsilon-greedy.
        """
        dealer = state.dealer_sum - 1
        player = state.player_sum - 1
        if np.random.rand() < (self.epsilon):
            # TODO
            print('Implement')
        else:
            action = np.random.choice(easyEnv.ACTIONS)

        return action

    def make_features(self, state, action):
        """Dealer's interval: dealer(s) = {[1, 4], [4, 7], [7, 10]}
         Player's interval: player(s) = {[1, 6], [4, 9], [7, 12], [10, 15],
         [13, 18], [16, 21]}
         Each binary feature has a value of 1 iff (s, a) lies within
         the cuboid of state-space corresponding to
         that feature, and the action corresponding to that feature.
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

        phi = phi.reshape(1, -1)

        return phi.astype(int)

    def linearly_combine_features(self):
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
                    # TODO
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

                # Initialize the state
                state = self.env.init_state()
                action = self.epsilon_greedy_policy(state)
                next_action = action

                while not state.terminal:

                    # Execute the action
                    next_state, reward = self.env.step(state, action)

                    if not next_state.terminal:
                        next_action = self.epsilon_greedy_policy(next_state)
                        #TODO

                    else:
                        # TODO

                    # Update = step-size × prediction error × feature value
                    self.eligibility = self.td_lambda * self.eligibility + \
                        self.make_features(state, action).reshape(-1, 1)
                    gradient = self.alpha * td_error * self.eligibility
                    # Adjust the weights
                    self.W += gradient

                    state = next_state
                    action = next_action

                if reward == 1:
                    self.player_wins += 1

                # TODO
                mse_term = np.sum((approximated_q - mc_agent_q)
                                  ** 2) / np.size(approximated_q)

                mse_per_lambdas[li, episode] = mse_term

                if episode % 1000 == 0 or episode+1 == self.num_episodes:
                    print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" % (
                        self.td_lambda, episode, mse_term, self.player_wins/(episode+1)))

            end_of_episode_mse[li] = mse_term

        return mse_per_lambdas, end_of_episode_mse

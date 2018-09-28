"""Q-learning, the simplest RL algorithm.

1.Initialize an array arbitrarily.
2.Choose actions based on Q, such that all
actions are taken in all states (infinitely often in the limit).
3.On each time step, change one element of the array:
    delta Q(S_t, A_t) = alpha * (R_t+1 + alpha * maxQ(S_t+1, a) - Q(St, At)
"""

import itertools
import matplotlib
import numpy as np

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, num_action):
    """Creates a policy, given a Q-function and epsilon.

    Args:
        Q: Dictionary map of state -> action.
        epsilon: The probability to select a random action.
        num_action: Number of available actions.

    Returns:
        policy_fn: Policy function.
    """
    def policy_fn(observations):
        action_set = np.ones(num_action, dtype=float) * epsilon/num_action
        best_act = np.argmax(Q[observations])
        action_set[best_act] += (1.0 - epsilon)
        return action_set

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """Q-learning algorithm. Off-policy TD control.
    This finds the the optimal greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths):
            Q is the optimal action-value function, a dictionary mapping
                state -> action values.
            stats is an EpisodeStats object with two numpy arrays for
                episode_lengths and episode_rewards.
    """

    # Action-value function Q. This is a nested dictionary that maps
    # action -> action-value.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for episode in range(num_episodes):

        state = env.reset()

        for t in itertools.count():
            # Take one step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)),
                                      p=action_probs)

            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t

            # TD update
            # Pick best action for the next state
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * \
                Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state
            env.render()

    return Q, stats


Q, stats = q_learning(env, 500)
plotting.plot_episode_stats(stats)

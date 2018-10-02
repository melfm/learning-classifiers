"""Sarsa - On-policy TD Control.

Example 6.5: Windy Gridworld.
"""

import numpy as np
import matplotlib.pyplot as plt


#######################
# Environment variables
#######################
# world height
WORLD_HEIGHT = 7
# world width
WORLD_WIDTH = 10
# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

# initial pose
START = [3, 0]
GOAL = [3, 7]


def step(state, action):

    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False


def episode(q_value):
    """Sarsa (on-policy TD control) for estimating Q.

    Initialize Q(s, a) arbitrarily except for Q(terminal,.) = 0

    Loop for each episode:
        Initialize S
        Choose A from S using Q(epsilon-greedy)
        Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q again
            Q(S_t, A_t) <- Q(S_t, A_t) + alpha*[R_t+1 + Q(S_t+1, A_t+1) \
                - Q(S_t, A_t)
            S <- S'; A <- A';
        until S is terminal

    Args:
        q_value: A matrix of dimension (H, W, 4) representing the state-action
        value function.

    Returns:
        time_elapsed: An Int, the total time steps in this episode.
    """
    time_elapsed = 0
    # initial state
    state = START

    # choose an epsilon-greedy action
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        # values for that state
        values = q_value[state[0], state[1], :]
        action = np.random.choice(
            [action for action, value in enumerate(values)
             if value == np.max(values)])

    # keep going until we reach the goal
    while state != GOAL:
        next_state = step(state, action)
        # choose an epsilon-greedy action again
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action for action, value in enumerate(values)
                 if value == np.max(values)])

        # Sarsa update
        q_value[state[0],
                state[1],
                action] += ALPHA * (REWARD + q_value[next_state[0],
                                                     next_state[1],
                                                     next_action] -
                                    q_value[state[0],
                                            state[1],
                                            action])
        state = next_state
        action = next_action
        time_elapsed += 1

    return time_elapsed


def run_and_visualize(episode_limit):

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    # The increasing slope of the graph shows that the goal
    # was reached more quickly over time (so as you go through
    # the episodes, they finish faster hence the steepness.
    plt.savefig('plots/figure_sarsa_windy_grid.png')
    plt.close()

    # display the optimal policy on the console
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


if __name__ == '__main__':

    episode_limit = 500
    run_and_visualize(episode_limit)

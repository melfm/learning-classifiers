"""Q1. Write an environment that implements the game Easy21.
We will be using this environment for model-free reinforcement learning,
and you should not explicitly represent the transition matrix for the
MDP. There is no discounting (γ = 1). You should treat the dealer’s moves as
part of the environment, i.e. calling step with a stick action will play out
the dealer’s cards and return the final reward and terminal state.
"""
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Actions: hit or stick
ACTION_HIT = 0
ACTION_STICK = 1
ACTIONS = [ACTION_HIT, ACTION_STICK]

# Initial policy for player
POLICY_PLAYER = np.zeros(22)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STICK
POLICY_PLAYER[21] = ACTION_STICK

# Policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STICK


# Draw a new card
def get_card():
    """Each draw from the deck results in a value between 1 and 10 (uniformly
    distributed) with a colour of red (probability 1/3) or black (probability
    2/3).
    """
    card = np.random.randint(1, 10)
    color_p = np.random.uniform(0, 1)
    if color_p < 1/3:
        color = 'RED'
    elif color_p >= 2/3:
        color = 'BLACK'
    return (card, color)


# Target policy of player
def target_policy_player(player_sum):
    return POLICY_PLAYER[player_sum]


def update_card_sum(new_card):

    card_value = 0
    if new_card[1] == 'BLACK':
        # Add the value of the card
        card_value = new_card[0]
    elif new_card[1] == 'RED':
        # Subtract the value of the card
        card_value = -new_card[0]
    else:
        raise ValueError('Received card with invalid colour!')

    return card_value


def step(input_state, action):
    """Takes as input a state 's' and an action 'a' (hit or stick), and returns the
    next state and reward r.

    Args:
        input_state: State s (dealer’s first card 1–10 and the player’s sum 1–21)
        action: Hit or stick

    Returns:
        next_state: A sample of next state s - this may be terminal if the game
            is finished.
        reward: An Int, Reward r.
    """

    #######################
    # Initialize the game
    #######################

    # Sum of player
    player_sum = 0
    # Sum of dealer
    dealer_sum = 0

    # Trajectory of player
    player_trajectory = []

    # Dealer status
    dealer_card1 = 0

    # At the start of the game, both the player and the dealer draw one black
    # card (fully observed)
    dealer_card1 = get_card()
    player_first_card = get_card()
    # Assume the initial draw is black and add the value
    player_sum += player_first_card[0]
    dealer_sum += dealer_card1[0]

    # Initialize player cards
    while player_sum < 12:
        card = get_card()
        player_sum += update_card_sum(card)

    state = [dealer_card1, player_sum]

    #######################
    # Game starts!
    #######################

    # Player's turn
    while True:
        action = target_policy_player(player_sum)

        player_trajectory.append([(player_sum, dealer_card1),
                                  action])

        if action == ACTION_STICK:
            # Next dealer starts taking turns
            break
        # If hit, draw a new card
        new_card = get_card()
        player_sum += update_card_sum(new_card)

        # Player busts
        if player_sum > 21:
            return state, -1, player_trajectory

    # Dealer's turn
    while True:
        # Get action based on current sum
        action = POLICY_DEALER[dealer_sum]
        if action == ACTION_STICK:
            break

        new_card = get_card()
        dealer_sum = update_card_sum(new_card)

        # Dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory

    # Compare the sum between player and dealer
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory

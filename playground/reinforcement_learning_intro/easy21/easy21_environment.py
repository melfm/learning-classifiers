"""Q1. Write an environment that implements the game Easy21.
We will be using this environment for model-free reinforcement learning,
and you should not explicitly represent the transition matrix for the
MDP. There is no discounting (γ = 1). You should treat the dealer’s moves as
part of the environment, i.e. calling step with a stick action will play out
the dealer’s cards and return the final reward and terminal state.
"""
import copy
import numpy as np

#######################
# Game Settings
#######################
# Actions: hit or stick
ACTION_HIT = 0
ACTION_STICK = 1
ACTIONS = [ACTION_HIT, ACTION_STICK]

COLOR_PROBS = {'red': 1/3, 'black': 2/3}
COLOR_VAL = {'red': -1, 'black': 1}


class State:
    """Maintains state of the game, including dealer cards, player
    cards, and whether the terminal state is reached.
    """

    def __init__(self, dealer_sum, player_sum, terminal=False):
        self.dealer_sum = dealer_sum
        self.player_sum = player_sum
        self.terminal = terminal

    def dealer_idx(self):
        # We want to exclude the 0 index
        return self.dealer_sum - 1

    def player_idx(self):
        return self.player_sum - 1


class Easy21Env:
    """Easy21 environment. The game rules are similar to Blackjack.
    """

    def __init__(self):
        self.player_value_count = 21
        self.dealer_value_count = 10
        self.action_count = 2
        self.dealer_sum = None
        self.player_sum = None

    def init_state(self):
        # The first draw is a black card
        dealer = self.draw_card()['value']
        player = self.draw_card()['value']
        return State(dealer, player)

    def draw_card(self, color=None):
        """Each draw from the deck results in a value between 1 and 10
        (uniformly distributed) with a colour of red (probability 1/3)
        or black (probability 2/3).
        """
        value = np.random.choice(range(1, self.dealer_value_count))
        if color is None:
            colors, probs = zip(*COLOR_PROBS.items())
            color = np.random.choice(colors, p=probs)
        return {'value': value, 'color': color}

    def bust(self, val):
        """Bust if value is less than or greater than 21.
        """
        return (val < 1 or val > 21)

    def step(self, state, action):
        """Takes an action (hit or stick), and returns the next state
            and reward.

        Args:
            state: Current state of type State object.
            action: Hit or stick

        Returns:
            next_state: A sample of next state s - this may be terminal if
                the game is finished.
            reward: Reward r.
        """

        next_state = copy.copy(state)
        reward = 0
        # Player's turn
        if action == ACTION_HIT:
            # Just draw a card, and check for bust. No reward here.
            card = self.draw_card()
            next_state.player_sum += COLOR_VAL[card['color']] * card['value']

            if self.bust(next_state.player_sum):
                next_state.terminal = True
                reward = -1

        # Dealer's turn
        elif action == ACTION_STICK:
            while not next_state.terminal:
                card = self.draw_card()
                # This is picking between adding/subtracting the value
                # based on the card color.
                next_state.dealer_sum += \
                    COLOR_VAL[card['color']] * card['value']
                if self.bust(next_state.dealer_sum):
                    next_state.terminal = True
                    reward = 1
                elif next_state.dealer_sum >= 17:
                    next_state.terminal = True
                    if next_state.dealer_sum > next_state.player_sum:
                        # Dealer won, negative reward
                        reward = -1
                    else:
                        # Player won, positive reward
                        reward = 1

        else:
            raise ValueError('Invalid choice of action.')

        return next_state, reward

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import easy21_environment as easyEnv
import monte_carlo_control as MC


def main(args):

    env = easyEnv.Easy21Env()

    agent = MC.MonteCarloAgent(env, num_episodes=args.num_episodes)

    agent.train()

    fig = plt.figure("N100")
    ax = fig.add_subplot(111, projection='3d')

    ax.clear()
    agent.plot_frame(ax)

    plt.title('V*')
    plt.ylabel('player sum', size=18)
    plt.xlabel('dealer sum', size=18)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes',
                        type=int,
                        default=100,
                        help='Number of episodes.')

    args = parser.parse_args()
    main(args)

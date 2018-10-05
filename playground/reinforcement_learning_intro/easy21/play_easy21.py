import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import easy21_environment as easyEnv
import monte_carlo_control as MC


def animate(frame, fig, ax, agent):

    episodes = agent.num_episodes

    # Train the agent inside this function, since we are
    # passing this to the `FuncAnimation`.
    agent.train()

    ax.clear()
    surf = agent.plot_frame(ax)
    mc_score = '%.1f' % (agent.player_wins/episodes*100.0)
    plt.title(
        'MC score:%s frame:%s episodes:%s ' %
        (mc_score, frame, episodes))
    fig.canvas.draw()

    return surf


def main(args):

    env = easyEnv.Easy21Env()

    agent = MC.MonteCarloAgent(env, num_episodes=args.num_episodes,
                               n0=args.n0)
    print('Agent training ....')

    fig = plt.figure('Monte-Carlo-Easy21')
    ax = fig.add_subplot(111, projection='3d')

    # Makes an animation by repeatedly calling a function, e.g. 30 frames.
    ani = animation.FuncAnimation(fig, animate, 10, fargs=(fig, ax, agent),
                                  repeat=True)

    ani.save('MC_Agent_N100.gif', writer='imagemagick', fps=3)

    """
    # Nomal plotting
    fig = plt.figure('Monte-Carlo-Easy21')
    ax = fig.add_subplot(111, projection='3d')

    ax.clear()
    agent.plot_frame(ax)

    plt.title('V*')
    plt.ylabel('player sum', size=18)
    plt.xlabel('dealer sum', size=18)

    plt.show()
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes',
                        type=int,
                        default=50000,
                        help='Number of episodes.')

    parser.add_argument('--n0',
                        type=int,
                        default=100,
                        help='Epsilon N constant.')

    args = parser.parse_args()
    main(args)

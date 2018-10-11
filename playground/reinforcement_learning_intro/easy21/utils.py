"""Utility functions.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from matplotlib import cm


def plot_frame(ax, env, v_func):

    X = np.arange(0, env.dealer_value_count, 1)
    Y = np.arange(0, env.player_value_count, 1)
    X, Y = np.meshgrid(X, Y)
    Z = v_func[X, Y]
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


def train_and_animate(fig_name, animation_frame, agent, num_episodes):

    fig = plt.figure(fig_name)
    ax = fig.add_subplot(111, projection='3d')

    # Makes an animation by repeatedly calling a function
    # 'animation_frame' times.
    ani = animation.FuncAnimation(fig, animate,
                                  animation_frame, fargs=(fig, ax, agent),
                                  repeat=True)

    gif_fig = fig_name + '_N' + str(num_episodes) + '.gif'
    ani.save(gif_fig, writer='imagemagick', fps=3)


def plot_lambda_vs_mse(td_lambdas, mse_per_lambda):

    plt.title('MSE - MC vs Sarsa (lambda)')
    plt.ylabel('MSE')
    plt.xlabel('Lambda')
    plt.plot(td_lambdas, mse_per_lambda)

    plt.show()

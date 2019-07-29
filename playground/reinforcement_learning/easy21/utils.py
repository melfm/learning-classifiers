"""Utility functions.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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
        cmap=plt.cm.viridis,
        linewidth=0,
        antialiased=False)
    return surf


def animate(frame, fig, ax, agent, env):
    """Plot animated function surface."""

    episodes = agent.num_episodes

    # Train the agent inside this function, since we are
    # passing this to the `FuncAnimation`.
    agent.train()

    ax.clear()
    surf = plot_frame(ax, env, agent.V)
    mc_score = '%.1f' % (agent.player_wins/episodes*100.0)
    plt.title(
        'MC score:%s frame:%s episodes:%s ' %
        (mc_score, frame, episodes))
    fig.canvas.draw()

    return surf


def train_and_animate(fig_name, animation_frame, agent, env, num_episodes):

    fig = plt.figure(fig_name)
    ax = fig.add_subplot(111, projection='3d')

    # Makes an animation by repeatedly calling a function
    # 'animation_frame' times.
    ani = animation.FuncAnimation(fig, animate,
                                  animation_frame, fargs=(fig, ax, agent, env),
                                  repeat=True)

    gif_fig = fig_name + '_N' + str(num_episodes) + '.gif'
    ani.save(gif_fig, writer='imagemagick', fps=3)


def plot_lambda_vs_mse(mse_per_lambda):

    fig = plt.figure()
    plt.title('Mean Squared Error Per Lambda')
    plt.ylabel('MSE')
    plt.xlabel('Lambda')
    td_lambdas = np.arange(0, 1.10, 0.1)
    plt.plot(td_lambdas, mse_per_lambda, '-o', linewidth=4.0)

    fig.savefig('TD-Sarsa-mse-per_lambda')


def plot_training_error(training_err, num_episodes, plot_name):

    training_step = np.arange(0, num_episodes, 1)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(training_step, training_err)

    plt.title('TD Sarsa training error')
    plt.ylabel('Training Error')
    plt.xlabel('Episode')

    plt_name = plot_name + '.png'
    fig.savefig(plt_name)


def plot_and_save(env, vstar, plot_name):
    """Plot static final surface."""

    X = np.arange(0, env.dealer_value_count, 1)
    Y = np.arange(0, env.player_value_count, 1)
    X, Y = np.meshgrid(X, Y)
    Z = vstar[X, Y]

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, linewidth=0.2)

    plt_name = plot_name + '.png'
    fig.savefig(plt_name)


def plot_mse_eps_per_lambda(mse_per_lambdas, episodes, agent_name='td-sarsa'):

    # https://stackoverflow.com/questions/45857465/create-a-2d-array-from-another-array-and-its-indices-with-numpy
    m, n = mse_per_lambdas.shape
    I, J = np.ogrid[:m, :n]
    out = np.empty((m, n, 3), dtype=mse_per_lambdas.dtype)
    out[..., 0] = I
    out[..., 1] = J
    out[..., 2] = mse_per_lambdas
    out.shape = (-1, 3)

    df = pd.DataFrame(out, columns=['lambda', 'Episode', 'MSE'])
    df['lambda'] = df['lambda'] / 10

    g = sns.FacetGrid(df, hue='lambda', size=8, legend_out=True)
    g = g.map(plt.plot, 'Episode', 'MSE').add_legend()

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Mean Squared Error per Episode')
    plot_name = 'MSE_for_all_lambdas_' + agent_name + '-' + str(episodes)
    g.savefig(plot_name)

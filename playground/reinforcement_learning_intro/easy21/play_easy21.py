import argparse
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d

import easy21_environment as easyEnv
import monte_carlo_control as MC
import td_learning_sarsa as TDS
import utils as util


def main(args):

    env = easyEnv.Easy21Env()

    agent_mode = args.agent
    fig_name = None

    if agent_mode == 'monte-carlo':
        print('Training Monte-Carlo Agent ...')
        mc_agent = MC.MonteCarloAgent(env, num_episodes=args.num_episodes,
                                      n0=args.n0)
        fig_name = 'Monte-Carlo-Easy21'

        util.train_and_animate(fig_name, 30, mc_agent, args.num_episodes)

    elif agent_mode == 'td-sarsa':
        print('Training TD-Sarsa Agent ...')
        mc_agent = MC.MonteCarloAgent(env, num_episodes=args.num_episodes,
                                      n0=args.n0)
        mc_agent.train()
        num_all_states = mc_agent.Q.shape[0] * mc_agent.Q.shape[1] * 2

        td_lambdas = np.arange(0, 1.10, 0.1)
        mse_per_lambda = []
        for lam in td_lambdas:

            sarsa_agent = TDS.SarsaAgent(env, num_episodes=args.num_episodes,
                                         n0=args.n0, td_lambda=lam)
            sarsa_agent.train()
            mse_term = np.sum(
                np.square(
                    (sarsa_agent.Q - mc_agent.Q))) / float(num_all_states)
            mse_per_lambda.append(mse_term)

        util.plot_lambda_vs_mse(td_lambdas, mse_per_lambda)

    else:
        raise ValueError('Invalid agent mode.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes',
                        type=int,
                        default=10000,
                        help='Number of episodes.')

    parser.add_argument('--n0',
                        type=int,
                        default=100,
                        help='Epsilon N constant.')

    parser.add_argument('--agent',
                        type=str,
                        default='monte-carlo',
                        help='Agent learning algorithm.')

    args = parser.parse_args()
    main(args)

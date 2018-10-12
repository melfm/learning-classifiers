import argparse
import numpy as np
import pickle

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

        if args.animate_plot:
            util.train_and_animate(fig_name, 100, mc_agent, env,
                                   args.num_episodes)
        else:
            mc_agent.train()

        plot_name = 'monte-carlo-surface-plot-' + str(args.num_episodes)
        util.plot_and_save(env, mc_agent.V, plot_name)

        if args.save_model:
            model_name = 'monte-carlo-model_' + \
                str(args.num_episodes) + '.pickle'
            pickle.dump(mc_agent.Q, open(model_name, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

    elif agent_mode == 'td-sarsa':
        print('Training TD-Sarsa Agent ...\n')
        print('Loading Monte-Carlo Agent ...\n')
        # Monte-Carlo iteration, used to save the model name
        mc_iter = 10000000
        model_name = 'monte-carlo-model_' + str(mc_iter) + '.pickle'
        mc_agent_q = pickle.load(open(model_name, 'rb'))

        if args.plot_tdsarsa_lambda:

            sarsa_agent = TDS.SarsaAgent(
                    env, num_episodes=args.num_episodes, n0=args.n0)
            mse_per_lambdas, end_of_episode_mse = sarsa_agent.train(mc_agent_q)
            util.plot_mse_eps_per_lambda(mse_per_lambdas)
            util.plot_lambda_vs_mse(end_of_episode_mse)


        if args.plot_train_err:
            num_episodes = 10000
            lam = 0

            sarsa_agent = TDS.SarsaAgent(env, num_episodes=num_episodes,
                                         n0=args.n0, td_lambda=lam)

            mse_per_lambdas, _ = sarsa_agent.train(mc_agent_q, run_single_lambda=True)

            training_err = np.squeeze(mse_per_lambdas)
            util.plot_training_error(training_err, num_episodes,
                                      'TD_training_err_lam_0')

            num_episodes = 10000
            lam = 1

            sarsa_agent = TDS.SarsaAgent(env, num_episodes=num_episodes,
                                         n0=args.n0, td_lambda=lam)
            mse_per_lambdas, _ = sarsa_agent.train(mc_agent_q, run_single_lambda=True)

            training_err = np.squeeze(mse_per_lambdas)
            util.plot_training_error(training_err, num_episodes,
                                      'TD_training_err_lam_1')

        if not args.plot_train_err and not args.plot_tdsarsa_lambda:
            raise ValueError(
                'You need to run TD-Lambda with at-least one of the options.')

    else:
        raise ValueError('Invalid agent mode.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes',
                        type=int,
                        default=1000,
                        help='Number of episodes.')

    parser.add_argument('--n0',
                        type=int,
                        default=100,
                        help='Epsilon N constant.')

    parser.add_argument('--agent',
                        type=str,
                        default='monte-carlo',
                        help='Agent learning algorithm.')

    parser.add_argument('--save_model',
                        type=bool,
                        default=True,
                        help='Store the agents Q-function.')

    parser.add_argument('--animate_plot',
                        type=bool,
                        default=False,
                        help='Create animated function surface.')

    parser.add_argument('--plot_tdsarsa_lambda',
                        type=bool,
                        default=True,
                        help='Plot MSE error against choice of lambda.')

    parser.add_argument('--plot_train_err',
                        type=bool,
                        default=False,
                        help='TD-Sarsa agent learning algorithm.')

    args = parser.parse_args()
    main(args)

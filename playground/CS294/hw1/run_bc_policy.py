#!/usr/bin/env python

"""Load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3.6 run_bc_policy.py --envname Humanoid-v2 --render --num_rollouts 20
"""
import argparse
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym

import bc_policy as bc
import network_params as par

#############################
# Network parameters
#############################
n_h1 = par.n_h1
n_h2 = par.n_h2
n_h3 = par.n_h3
batch_size = par.batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('Loading and building expert policy...')

    with open('rollouts/' + args.envname + '.pkl', 'rb') as f:
        data = pickle.load(f)

        rollouts = data['observations']
        trajec_size = rollouts.shape
        input_size = trajec_size[1]
        data_actions = data['actions']
        output_size = data_actions.shape[2]

    x, y = bc.placeholder_inputs(None, input_size, output_size, batch_size)
    policy_fn = bc.inference(x, input_size, output_size, n_h1, n_h2, n_h3)

    saver = tf.train.Saver()

    with tf.Session():
        tf_util.initialize()
        saver.restore(tf_util.get_session(), 'trainedNN/' + args.envname)

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('Rollout iteration ', i)
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done:
                action = tf_util.get_session().run([policy_fn],
                                                   feed_dict={x: obs[None, :]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action[0])
                total_reward += r
                steps += 1
                if args.render:
                    env.render()
                    if steps % 100 == 0:
                        print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
            returns.append(total_reward)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()

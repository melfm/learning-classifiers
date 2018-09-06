#!/usr/bin/env python

"""DAgger - Dataset Aggregation.
Example usage:
    python3.6 run_dagger.py --envname Humanoid-v2 --render --num_rollouts 20
"""
import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym

import load_policy
import tf_util
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
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--dagger_iter', type=int, default=10)
    args = parser.parse_args()

    expert_policy_file = 'experts/Humanoid-v2.pkl'
    expert_policy_file = 'experts/' + args.envname + '.pkl'
    expert_policy_fn = load_policy.load_policy(expert_policy_file)

    ##############################################################
    # Load expert data rollouts
    ##############################################################
    print('Loading and building expert policy...')

    with open('rollouts/' + args.envname + '.pkl', 'rb') as f:
        data = pickle.load(f)

        obs_data = data['observations']
        trajec_size = obs_data.shape
        trajec_length = trajec_size[0]
        input_size = trajec_size[1]
        action_data = data['actions']
        output_size = action_data.shape[2]
        total_batch = int(trajec_length/batch_size)

    #####################################################################
    # Set up the network for the imitation learning policy function
    #####################################################################
    x, y = bc.placeholder_inputs(None, input_size, output_size, batch_size)
    policy_fn = bc.inference(x, input_size, output_size, n_h1, n_h2, n_h3)
    loss_l2 = bc.l2loss(policy_fn, y)
    train_op = bc.train(loss_l2, args.learning_rate)

    #####################################################################
    # Run DAgger algorithm
    #####################################################################
    with tf.Session() as sess:
        tf_util.initialize()
        total_mean = []
        total_std = []
        total_train_size = []

        # DAgger Loop
        for i_dagger in range(args.dagger_iter):
            print('DAgger iteration ', i_dagger)
            #################################
            # 1. Train a policy from the data
            #################################
            for epoch in range(args.num_epochs):
                total_loss = 0
                for i in range(total_batch):
                    # Process by batch
                    x_feed = obs_data[i*batch_size:(i+1)*batch_size]
                    y_feed = action_data[i *
                                         batch_size:(i +
                                                     1) *
                                         batch_size].reshape(batch_size,
                                                             output_size)
                    feed_dict = {
                        x: x_feed,
                        y: y_feed,
                    }
                    _, train_loss = sess.run(
                        [train_op, loss_l2],
                        feed_dict=feed_dict)
                    total_loss += train_loss / total_batch

                if epoch % 10 == 0:
                    print(
                        "Epoch: ",
                        "%04d" %
                        (epoch),
                        "loss= ",
                        "{:.9f}".format(total_loss))

            print('Finished training the network.')
            #################################
            # 2. Run the policy
            #################################
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []

            for i in range(args.num_rollouts):
                print('Rollout iter ', i)

                obs = env.reset()
                done = False
                total_reward = 0
                steps = 0
                while not done:
                    action = sess.run([policy_fn],
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
            print('Return-Mean ', np.mean(returns))
            print('Return-Std ', np.std(returns))
            #################################
            # 3. Expert Labelling
            #################################
            action_new = []

            for i_label in range(len(observations)):
                action_new.append(expert_policy_fn(
                    observations[i_label][None, :]))

            train_data_size = obs_data.shape[0]
            #################################
            # 4. Aggregate
            #################################
            obs_data = np.concatenate((obs_data,
                                       np.array(observations)),
                                      axis=0)

            action_data = np.concatenate((action_data,
                                          np.array(action_new)),
                                         axis=0)
            # Store mean return and std
            total_mean = np.append(total_mean, np.mean(returns))
            total_std = np.append(total_std, np.std(returns))
            total_train_size = np.append(total_train_size, train_data_size)

    dagger_results = {
        'means': total_mean,
        'stds': total_std,
        'train_size': total_train_size}

    with open('DAgger_results.pkl', 'wb') as fp:
        pickle.dump(dagger_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('means ', total_mean)
    print('stds ', total_std)
    print('train_size', total_train_size)


if __name__ == '__main__':
    main()

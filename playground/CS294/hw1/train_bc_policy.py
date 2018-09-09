""" This trains the Feedforward neural network using the generated rollouts.

Example usage:
    python3.6 train_bc_policy.py Humanoid-v2 --num_epochs 500
"""

import argparse
import tensorflow as tf
import pickle

import bc_policy as bc
import network_params as par

n_h1 = par.n_h1
n_h2 = par.n_h2
n_h3 = par.n_h3
batch_size = par.batch_size


def train_network(args):
    with open('rollouts/' + args.name + '.pkl', 'rb') as f:
        data = pickle.load(f)

    observations = data['observations']
    trajec_size = observations.shape
    trajec_length = trajec_size[0]
    input_size = trajec_size[1]
    actions = data['actions']
    output_size = actions.shape[2]
    total_batch = int(trajec_length/batch_size)

    x, y = bc.placeholder_inputs(None, input_size, output_size, batch_size)

    logits = bc.inference(x, input_size, output_size, n_h1, n_h2, n_h3)
    loss = bc.l2loss(logits, y)

    train_op = bc.train(loss, args.learning_rate)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('trainedNN/' + args.name +
                                         '/train_summary', sess.graph)

    sess.run(init)
    for epoch in range(args.num_epochs):
        total_loss = 0
        for i in range(total_batch):
            feed_dict = bc.fill_feed_dict(
                x, y, data, i, output_size, batch_size)
            _, train_loss, summary = sess.run([train_op, loss, merged],
                                              feed_dict=feed_dict)
            total_loss += train_loss / total_batch
            train_writer.add_summary(summary, i)

        if epoch % 10 == 0:
            print(
                "Epoch: ",
                "%04d" %
                (epoch),
                "loss= ",
                "{:.9f}".format(total_loss))

        saver.save(sess, 'trainedNN/' + args.name)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='Humanoid-v2')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--firsttime', type=int)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4)
    args = parser.parse_args()

    train_network(args)

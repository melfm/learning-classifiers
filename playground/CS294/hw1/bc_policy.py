""" This should just be a Feedforward neural network setup."""

import numpy as np
import tensorflow as tf


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.random_uniform(shape=shape, minval=-np.sqrt(6.0/sum(shape)),
                                maxval=np.sqrt(6.0/sum(shape)))
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Make a simple neural net layer.
    """
    # Adding a name scope to group layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def l2loss(pred, target):
    square_diff = tf.nn.l2_loss(pred - target)
    tf.summary.scalar('L2_loss', square_diff)
    return square_diff


def train(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trainstep = optimizer.minimize(loss)
    return trainstep


def variable_summaries(var):
    """Attach summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def inference(x, nin, nout, n_h1, n_h2, n_h3):
    layer1 = nn_layer(x, nin, n_h1, 'layer1')
    layer2 = nn_layer(layer1, n_h1, n_h2, 'layer2')
    layer3 = nn_layer(layer2, n_h2, n_h3, 'layer3')
    with tf.name_scope('outlayer'):
        with tf.name_scope('weights'):
            W = weight_variable([n_h3, nout])
        with tf.name_scope('biases'):
            b = bias_variable([nout])
    out = tf.matmul(layer3, W) + b

    return out


def placeholder_inputs(size, nin, nout, batch_size):
    x_placeholder = tf.placeholder(tf.float32, shape=(size,
                                                      nin))
    y_placeholder = tf.placeholder(tf.float32, shape=(size,
                                                      nout))
    return x_placeholder, y_placeholder


def fill_feed_dict(x, y, data, i, nout, batch_size):

    x_feed = data['observations'][i*batch_size:(i+1)*batch_size]
    y_feed = data['actions'][i * batch_size:(i + 1) *
                             batch_size].reshape(batch_size, nout)
    feed_dict = {
        x: x_feed,
        y: y_feed,
    }
    return feed_dict

from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        #######################################################################
        # Initialize the weights and biases of the two-layer net. Weights
        # should be initialized from a Gaussian centered at 0.0 with
        # standard deviation equal to weight_scale, and biases should be
        # initialized to zero. All weights and biases should be stored in the
        # dictionary self.params, with first layer weights
        # and biases using the keys 'W1' and 'b1' and second layer
        # weights and biases using the keys 'W2' and 'b2'.
        #######################################################################
        D = input_dim
        H = hidden_dim
        self.params['W1'] = weight_scale * np.random.randn(D, H)
        self.params['b1'] = np.zeros(H)
        self.params['W2'] = weight_scale * np.random.randn(H, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ########################################################################
        # Implement the forward pass for the two-layer net, computing the
        # class scores for X and storing them in the scores variable.
        ########################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        D = np.prod(X.shape[1:])

        X = X.reshape(-1, D)

        layer_1 = np.dot(X, W1) + b1
        hidden_layer = np.maximum(0, layer_1)
        layer_2 = np.dot(hidden_layer, W2) + b2

        scores = layer_2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ########################################################################
        # Implement the backward pass for the two-layer net. Store the loss
        # in the loss variable and gradients in the grads dictionary. Compute
        # data loss using softmax, and make sure that grads[k] holds the
        # gradients for self.params[k]. Don't forget to add L2 regularization!
        #
        # NOTE: To ensure that your implementation matches ours and you pass the
        # automated tests, make sure that your L2 regularization includes a
        # factor of 0.5 to simplify the expression for the gradient.
        ########################################################################
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        data_loss = -np.sum(log_probs[np.arange(N), y]) / N

        # Note that the 0.5 is a trick that simplifies the gradient expression
        # Altho it causes the losses to be slightly different
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        loss = data_loss + reg_loss

        # Compute the gradient on scores
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.ones((N)).dot(dscores)

        # Backprop into the hidden layer
        # We had np.dot(hidden_l, W2) + b2
        dhidden_l = np.dot(dscores, W2.T)
        dhidden_l[hidden_layer <= 0] = 0
        # Backprop to the first layer
        dW1 = np.dot(X.T, dhidden_l)
        db1 = np.ones((N)).dot(dhidden_l)

        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ########################################################################
        # Initialize the parameters of the network, storing all values in
        # the self.params dictionary. Store weights and biases for the first layer
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be
        # initialized from a normal distribution centered at 0 with standard
        # deviation equal to weight_scale. Biases should be initialized to zero.
        #
        # When using batch normalization, store scale and shift parameters for the
        # first layer in gamma1 and beta1; for the second layer use gamma2 and
        # beta2, etc. Scale parameters should be initialized to ones and shift
        # parameters should be initialized to zeros.
        ########################################################################

        layer_counter = 1
        next_input_dim = input_dim
        for layer in range(self.num_layers):
            if layer == 0:
                hidden_dim = hidden_dims[layer]
                self.params['W1'] = weight_scale * \
                    np.random.randn(next_input_dim, hidden_dim)
                self.params['b1'] = np.zeros(hidden_dim)
                next_input_dim = hidden_dim
            elif layer == self.num_layers - 1:
                # final layer
                layer_counter += 1
                W_name = 'W' + str(layer_counter)
                b_name = 'b' + str(layer_counter)
                self.params[W_name] = weight_scale * \
                    np.random.randn(next_input_dim, num_classes)
                self.params[b_name] = np.zeros(num_classes)

            else:
                layer_counter += 1
                W_name = 'W' + str(layer_counter)
                b_name = 'b' + str(layer_counter)
                hidden_dim = hidden_dims[layer]
                self.params[W_name] = weight_scale * \
                    np.random.randn(next_input_dim, hidden_dim)
                self.params[b_name] = np.zeros(hidden_dim)
                next_input_dim = hidden_dim

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ########################################################################
        # Implement the forward pass for the fully-connected net, computing
        # the class scores for X and storing them in the scores variable.
        #
        # When using dropout, you'll need to pass self.dropout_param to each
        # dropout forward pass.
        #
        # When using batch normalization, you'll need to pass self.bn_params[0]
        # to the forward pass for the first batch normalization layer, pass
        # self.bn_params[1] to the forward pass for the second batch
        # normalization layer, etc.
        ########################################################################

        hidden_layers = {}
        layer_counter = 0
        weight_sums = 0

        for layer in range(self.num_layers):
            if layer == 0:
                W1, b1 = self.params['W1'], self.params['b1']
                D = np.prod(X.shape[1:])
                X = X.reshape(-1, D)
                hidden_layer_1 = np.maximum(0, np.dot(X, W1) + b1)
                hidden_layers['hl_1'] = hidden_layer_1

                layer_counter = 1

                weight_sums += np.sum(W1 * W1)

            elif layer == self.num_layers - 1:
                # final layer
                layer_counter += 1
                W_name = 'W' + str(layer_counter)
                b_name = 'b' + str(layer_counter)
                W = self.params[W_name]

                previous_layer_n = 'hl_' + str(layer_counter - 1)
                previous_layer = hidden_layers[previous_layer_n]

                net_out = np.dot(previous_layer, W) + self.params[b_name]
                current_layer_n = 'hl_' + str(layer_counter)
                hidden_layers[current_layer_n] = net_out
                scores = net_out

                weight_sums += np.sum(W * W)

            else:
                layer_counter += 1
                W_name = 'W' + str(layer_counter)
                b_name = 'b' + str(layer_counter)
                W = self.params[W_name]

                previous_layer_n = 'hl_' + str(layer_counter - 1)
                previous_layer = hidden_layers[previous_layer_n]

                Z = np.dot(previous_layer, W) + self.params[b_name]
                Z_relu = np.maximum(0, Z)
                current_layer_n = 'hl_' + str(layer_counter)
                hidden_layers[current_layer_n] = Z_relu

                weight_sums += np.sum(W * W)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ########################################################################
        # Implement the backward pass for the fully-connected net. Store the
        # loss in the loss variable and gradients in the grads dictionary.
        # Compute data loss using softmax, and make sure that grads[k] holds the
        # gradients for self.params[k]. Don't forget to add L2 regularization!
        #
        # When using batch/layer normalization, you don't need to regularize the
        # scale and shift parameters.
        #
        # NOTE: To ensure that your implementation matches ours and you pass the
        # automated tests, make sure that your L2 regularization includes a
        # factor of 0.5 to simplify the expression for the gradient.
        ########################################################################
        N = X.shape[0]
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        data_loss = -np.sum(log_probs[np.arange(N), y]) / N

        reg_loss = 0.5 * self.reg * weight_sums

        loss = data_loss + reg_loss

        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        previous_layer = self.num_layers
        previous_dlayer = None
        for layer in range(self.num_layers, 0, -1):
            W_name = 'W' + str(layer)
            b_name = 'b' + str(layer)

            if layer == self.num_layers:
                # Last layer
                previous_layer_n = 'hl_' + str(layer - 1)
                dW = np.dot(hidden_layers[previous_layer_n].T,
                            dscores)
                db = np.ones((N)).dot(dscores)

                dW += self.reg * self.params[W_name]
                grads[W_name] = dW
                grads[b_name] = db

                dhidden_l = np.dot(dscores, self.params[W_name].T)
                dhidden_l[hidden_layers[previous_layer_n] <= 0] = 0
                previous_dlayer = dhidden_l

                previous_layer -= 1

            elif layer == 1:
                # First layer
                dW1 = np.dot(X.T, previous_dlayer)
                db1 = np.ones((N)).dot(previous_dlayer)
                dW1 += self.reg * self.params[W_name]
                grads['W1'] = dW1
                grads['b1'] = db1

            else:
                previous_layer_n = 'hl_' + str(layer - 1)
                dW = np.dot(hidden_layers[previous_layer_n].T,
                            previous_dlayer)
                db = np.ones((N)).dot(previous_dlayer)
                dW_n = W_name
                db_n = 'b' + str(layer)
                dW += self.reg * self.params[W_name]
                grads[dW_n] = dW
                grads[db_n] = db

                dhidden_l = np.dot(previous_dlayer, self.params[W_name].T)
                dhidden_l[hidden_layers[previous_layer_n] <= 0] = 0
                previous_dlayer = dhidden_l

                previous_layer -= 1

        return loss, grads

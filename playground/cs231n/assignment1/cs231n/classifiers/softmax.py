import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # Compute the softmax loss and its gradient using explicit loops.
    # Store the loss in loss and the gradient in dW. If you are not careful
    # here, it is easy to run into numeric instability. Don't forget the
    # regularization!
    ##########################################################################
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        # Shift so that the highest value is 0
        scores -= np.max(scores)
        scores_exp = np.exp(scores)
        # Target class score
        correct_class_score = scores[y[i]]
        score_total = np.sum(scores_exp, axis=0)

        scores_normalized = scores_exp / score_total
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += -X[i] + (scores_normalized[j]) * X[i]
            else:
                dW[:, j] += scores_normalized[j] * X[i]

        scores_exp = np.exp(correct_class_score)
        scores_exp_sum = np.sum(np.exp(scores))
        loss += -np.log(scores_exp / scores_exp_sum)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # Compute the softmax loss and its gradient using no explicit loops.
    # Store the loss in loss and the gradient in dW. If you are not careful
    # here, it is easy to run into numeric instability. Don't forget the
    # regularization!
    ##########################################################################
    num_train = X.shape[0]

    scores = X.dot(W)
    max_scores = np.max(scores, axis=1)
    scores -= max_scores[:, np.newaxis]

    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1)

    correct_class_score = scores[np.arange(num_train), y]
    correct_class_score_exp = np.exp(correct_class_score)

    scores_normalized = scores_exp / scores_exp_sum[:, np.newaxis]
    # Add the (1 - ...) for the case where y[i] == j
    scores_normalized[np.arange(num_train), y] -= 1.0
    dW = X.T.dot(scores_normalized)

    loss = -np.sum(np.log(correct_class_score_exp / scores_exp_sum))

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW

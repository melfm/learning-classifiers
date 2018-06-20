import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on
    minibatches of N examples.

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):

        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                # This is computing the derivative as the loss is being
                # calculated. The intuition for this is as follows:
                # Everytime loss is computed for X[i] and its nonzero,
                # you should move your weight vector for the incorrect class
                # (j != y[i]) away by X[i] - And move the weights or hyperplane
                # for the correct class (j == y[i]) near X[i].
                # This way dW[y[i]] tries to come nearer to X[i].
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    delta = 1

    ##########################################################################
    # Implement a vectorized version of the structured SVM loss, storing the
    # result in loss.
    ##########################################################################
    scores = X.dot(W)
    # This needs to access the scores per training instance
    # So we use np.arange to index along the instances

    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(
        0, scores - correct_class_scores[:, np.newaxis] + delta)
    margins[np.arange(num_train), y] = 0

    mask = np.zeros((margins.shape))
    mask[margins > 0] = 1

    # Sum along each column, then take *negative* values and assign
    # to y_i component.
    row_sum = np.sum(mask, axis=1)
    # This is equivalent to the step -=X[i] we were doing before!
    mask[np.arange(num_train), y] = -row_sum
    # This then takes care of the j component.
    dW = X.T.dot(mask)
    loss += np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    ##########################################################################
    # Implement a vectorized version of the gradient for the structured SVM
    # loss, storing the result in dW.
    #
    # Hint: Instead of computing the gradient from scratch, it may be easier
    # to reuse some of the intermediate values that you used to compute the
    # loss.
    ##########################################################################

    return loss, dW

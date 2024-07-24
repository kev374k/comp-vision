from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = X.shape[0]
    for i in range(n):
        # X[:, i] represents an individual batch of data
        wx = np.dot(X[i, :], W)

        # removing numeric instability
        wx -= np.max(wx)

        sum_wx = np.sum(np.exp(wx))

        # when deriving the softmax function, we get loss = -fyi + log(sum(efj))
        loss += -wx[y[i]] + np.log(sum_wx)
        
        for j in range(W.shape[1]):
            # For the derivative, we see that when fyi = fj, the result is (p - 1) * x, otherwise it is just p * x
            p = np.exp(wx[j]) / sum_wx
            dW[:, j] += (p - (j == y[i])) * X[i, :]

    # divide by the number of examples
    loss /= n
    dW /= n

    # regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X @ W
    scores -= np.max(scores, axis = 1, keepdims = True)
    out = np.exp(scores)
    sum_wx = out / np.sum(out, axis = 1, keepdims = True)

    loss -= np.sum(np.log(sum_wx[np.arange(num_train), y]))
    loss /= num_train
    loss += 0.5 ** reg * np.sum(W**2)

    dout = np.copy(sum_wx)

    # (N, C)
    dout[np.arange(num_train), y] -= 1

    # (D, N) x (N, C)
    dW = X.T @ dout
    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

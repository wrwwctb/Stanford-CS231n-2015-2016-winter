import numpy as np
from random import shuffle
from scipy.misc import logsumexp

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
  # loss = 0.0
  # dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  S = np.dot(X, W)
  dS_dW = np.einsum('im,nj->ijmn', X, np.eye(C))  # 30324167
  rangeN = range(N)
  numerator = -S[rangeN, y].reshape((-1, 1))
  denominator = logsumexp(S, axis=1, keepdims=True)
  Li = numerator + denominator
  loss = Li.mean() + .5 * reg * np.sum(W**2)  # dataloss + regloss

  dLi_dS = np.zeros((N, C))
  dLi_dS[rangeN, y] = -1  # numerator part # dnumerator_dW = -dS_dW[rangeN, y, :, :]  # 3rd order tensor sampled from dS_dW
  dLi_dS += np.exp(S-denominator)  # denominator part
  dLi_dW = np.einsum('ij,ijmn->imn', dLi_dS, dS_dW)  # LOOP IS IN einsum
  dW = np.sum(dLi_dW, axis=0)/N + reg * W  # d dataloss/dW + d regloss/dW

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  # loss = 0.0
  # dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  scores = np.dot(X, W)
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  loss = -np.log(prob[range(N), y])
  loss = loss.mean()
  loss += .5 * reg * (np.sum(W**2))
    
  dscores = np.zeros_like(scores)
  dscores[range(N), y] = -1
  dscores += prob
  dscores /= N
  dW = np.dot(X.T, dscores) + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


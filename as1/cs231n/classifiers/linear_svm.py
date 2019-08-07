import numpy as np
from random import shuffle

def svm_loss_naive_first_success(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dmax_dscores = np.zeros((num_train, num_classes))
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dmax_dscores[i, j] = 1
        dmax_dscores[i, y[i]] += -1  #################################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dtloss_dtloss = 1  # tloss = loss + reg

  dtloss_dterm2 = 1
  dterm2_dW = reg * W
  dtloss_dW1 = dtloss_dterm2 * dterm2_dW

  dtloss_dterm1 = 1
  dterm1_dsumLi = 1/num_train
  dtloss_dsumLi = dtloss_dterm1 * dterm1_dsumLi
  dsumLi_dLi = 1
  dtloss_dLi = dtloss_dsumLi * dsumLi_dLi
  dLi_dmax = 1
  dtloss_dmax = dtloss_dLi * dLi_dmax
  dtloss_dscores = dtloss_dmax * dmax_dscores  # (num_train, num_classes)
  dscores_dW = X.T
  dtloss_dW2 = np.dot(dscores_dW, dtloss_dscores)

  dW = dtloss_dW1 + dtloss_dW2

  return loss, dW


def svm_loss_naive(W, X, y, reg):
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_pix = X.shape[1]
  loss = 0.0
  dloss_dW = np.zeros(W.shape)
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    dscores_dW = np.zeros((num_pix, num_classes, num_classes))  # for each score si, dsi/dW is an array: [:, :, i]
    for j in range(num_classes):
        dscores_dW[:, j, j] = X[i]
    dloss_dscores = np.zeros((1, num_classes))
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        loss += margin
        dloss_dscores[0, j] += 1
        dloss_dscores[0, y[i]] -= 1
    dloss_dW += np.squeeze(np.inner(dloss_dscores, dscores_dW))

  loss /= num_train
  dloss_dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dloss_dW += reg * W

  return loss, dloss_dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  # loss = 0.0
  # dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  S1 = np.dot(X, W)
  S2 = S1 - S1[range(num_train), y].reshape(-1, 1) + 1
  S2[range(num_train), y] = 0
  S2 = np.maximum(S2, 0)
  loss = np.sum(S2)/num_train + .5*reg*np.sum(W**2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dS2 = 1/num_train
  S2p = (S2 > 0)*1  # pattern. records contribution to loss of each entry in S1
  S2p[range(num_train), y] = -np.sum(S2p, axis=1)
  dS1 = dS2 * S2p
  dW = np.dot(X.T, dS1) + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

# old, correct attempt
#  num_train = X.shape[0]
#  S = np.dot(X, W)
#  # dS/dW = 4d tensor. seek alternative
#  correct_scores = S[range(num_train), y]
#  Sbar = S - correct_scores.reshape(num_train, 1) + 1
#  Sbar[range(num_train), y] = 0
#  Sbar = np.maximum(Sbar, 0)
#  sumSbar = np.sum(Sbar)
#  loss = sumSbar/num_train + .5 * reg * np.sum(W**2)
#
#  #Spattern = (Sbar > 0) * 1  # gives a runtime warning during loop validation
#  Spattern = Sbar.astype('bool') * np.ones(1, dtype=X.dtype)[0]#1#np.float128(1)
#  Spattern[range(num_train), y] = -np.sum(Spattern, axis=1)
#  dsumSbar_dW = np.dot(X.T, Spattern)
#  dW = dsumSbar_dW/num_train + reg * W

  return loss, dW

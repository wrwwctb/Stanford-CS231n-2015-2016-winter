import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pad = int(np.floor(filter_size/2))  # see loss()
    C, H, W = input_dim
    Hf = Wf = filter_size
    F = num_filters
    M = hidden_dim
    stride = 1  # see loss()
    assert (H-Hf+2*pad) % stride == 0, 'haha'
    assert (W-Wf+2*pad) % stride == 0, 'haha'
    Hr = (H-Hf+2*pad)//stride+1
    Wr = (W-Wf+2*pad)//stride+1
    Hmp = Hr//2  # max pool is 2x2, fixed
    Wmp = Wr//2
    D = F * Hmp * Wmp
    Cl = num_classes
    W1 = np.random.normal(0, weight_scale, (F, C, Hf, Wf))
    b1 = np.zeros(F)
    W2 = np.random.normal(0, weight_scale, (D, M))
    b2 = np.zeros(M)
    W3 = np.random.normal(0, weight_scale, (M, Cl))
    b3 = np.zeros(Cl)

    self.params['W1'] = W1
    self.params['W2'] = W2
    self.params['W3'] = W3
    self.params['b1'] = b1
    self.params['b2'] = b2
    self.params['b3'] = b3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}  # was single /

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    '''
    outC, cacheC = conv_forward_fast(X, W1, b1, conv_param)
    outR1, cacheR1 = relu_forward(outC)
    outMP, cacheMP = max_pool_forward_fast(outR1, pool_param)
    outA1, cacheA1 = affine_forward(outMP, W2, b2)
    outR2, cacheR2 = relu_forward(outA1)
    scores, cacheA2 = affine_forward(outR2, W3, b3)
    '''
    outMP, cacheMP = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    outR2, cacheR2 = affine_relu_forward(outMP, W2, b2)
    scores, cacheA2 = affine_forward(outR2, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dA2 = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    '''
    dR2, dW3, db3 = affine_backward(dA2, cacheA2)
    dA1 = relu_backward(dR2, cacheR2)
    dMP, dW2, db2 = affine_backward(dA1, cacheA1)
    dR1 = max_pool_backward_fast(dMP, cacheMP)
    dC = relu_backward(dR1, cacheR1)
    _, dW1, db1 = conv_backward_fast(dC, cacheC)
    '''
    dR2, dW3, db3 = affine_backward(dA2, cacheA2)
    dMP, dW2, db2 = affine_relu_backward(dR2, cacheR2)
    _, dW1, db1 = conv_relu_pool_backward(dMP, cacheMP)
    
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


class CustomNet(object):
  """
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  # 7   0   0   2   3   0   0   2   200 0   0   0
  # 32  0   0   2   64  0   0   2   0   0   0   0
  # c   b   r   p   c   b   r   p   f   b   r   f   s
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pad = int(np.floor(filter_size/2))  # see loss()
    C, H, W = input_dim
    Hf = Wf = filter_size
    F = num_filters
    M = hidden_dim
    stride = 1  # see loss()
    assert (H-Hf+2*pad) % stride == 0, 'haha'
    assert (W-Wf+2*pad) % stride == 0, 'haha'
    Hr = (H-Hf+2*pad)//stride+1
    Wr = (W-Wf+2*pad)//stride+1
    Hmp = Hr//2  # max pool is 2x2, fixed
    Wmp = Wr//2
    D = 2*F * Hmp//2 * Wmp//2
    Cl = num_classes
    W1 = np.random.normal(0, weight_scale, (F, C, Hf, Wf))
    b1 = np.zeros(F)
    Wa = np.random.normal(0, weight_scale, (2*F, F, 3, 3))
    ba = np.zeros(2*F)
    W2 = np.random.normal(0, weight_scale, (D, M))
    b2 = np.zeros(M)
    W3 = np.random.normal(0, weight_scale, (M, Cl))
    b3 = np.zeros(Cl)

    self.params['W1'] = W1
    self.params['Wa'] = Wa
    self.params['W2'] = W2
    self.params['W3'] = W3
    self.params['b1'] = b1
    self.params['ba'] = ba
    self.params['b2'] = b2
    self.params['b3'] = b3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    Wa, ba = self.params['Wa'], self.params['ba']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}  # was single /
    conv_parama = {'stride': 1, 'pad': 1} 
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    outP1, cacheP1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    outP2, cacheP2 = conv_relu_pool_forward(outP1, Wa, ba, conv_parama, pool_param)
    outR2, cacheR2 = affine_relu_forward(outP2, W2, b2)
    scores, cacheA2 = affine_forward(outR2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dA2 = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    
    dR2, dW3, db3 = affine_backward(dA2, cacheA2)
    dP2, dW2, db2 = affine_relu_backward(dR2, cacheR2)
    dP1, dWa, dba = conv_relu_pool_backward(dP2, cacheP2)
    _, dW1, db1 = conv_relu_pool_backward(dP1, cacheP1)
    
    dW1 += self.reg * W1
    dWa += self.reg * Wa
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    
    grads['W1'] = dW1
    grads['Wa'] = dWa
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['b1'] = db1
    grads['ba'] = dba
    grads['b2'] = db2
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
pass

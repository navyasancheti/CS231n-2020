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
    num_train= X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores=X.dot(W) 
    for i in range(num_train):
      den_sum=0
      max_score=np.max(scores[i,:])
      arr=scores[i].copy()
      arr -=max_score

      den_sum = np.sum(np.exp(arr))
      correct_score = arr[y[i]]
      prob = np.exp(correct_score) / den_sum
      loss += -np.log(prob)

      for j in range(scores.shape[1]):
        Sj = np.exp(scores[i,j]-np.max(scores[i,:]))/den_sum        

        if j==y[i]:
            dW[:,j] += X[i] * (Sj - 1)
        else:
            dW[:,j] += X[i] * (Sj)

    loss/= num_train
    loss += 0.05*reg *np.sum(W*W)

    dW /=num_train
    dW += reg*W


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
    num_train=X.shape[0]
    num_classes=W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores= X.dot(W)
    max_arr=np.max(scores,axis=1)

    scores -= max_arr[:,None]

    correct_arr = np.matrix(scores[np.arange(num_train),y]).T
    den_sum = np.sum(np.exp(scores),axis=1)

    P = np.exp(correct_arr)/den_sum
    loss = - np.mean(np.log(P))

    S = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:,None] # (N, C)
    S_corr = S - np.equal(np.arange(num_classes), y[:,None]) # (N, C)
    dW += np.dot(X.T, S_corr) # (D, C)
    dW /= num_train

    dW += reg*W
    loss += 0.05*reg*np.sum(W*W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

import numpy as np

def compute_mse(y, tx, w):
    """
    calculates MSE loss 

    :param y: 
    :type y: np array
    :param tx: extended feature matrix
    :type tx: numpy 2D array
    :param w: regression weight parameters
    :type w: numpy array
    :return: loss value
    :rtype: float
    """
    # convert row arrays to matrices in correct shape
    # numpy can actually implicitly do this, but to keep things readable we prefer to do it manually 
    w = w.reshape((-1,1))
    y = y.reshape((-1,1))
    
    # calculate e
    e =  y- np.dot(tx,w)
    
    #calculate loss
    loss =  np.dot(e.T,e)[0,0]/ 2 / y.shape[0]
    
    return loss

def compute_ridge_loss(y , tx ,w , lambda_):
    """
    implements the ridge regression loss function
    :param y: labels
    :type y: numpy array
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each column a feature
    :type tx: numpy 2D array
    :param lambda_: Regularizer
    :type lambda_: float64

    :return: loss value
    :rtype:  float64
    
    """
    return compute_mse(y, tx, w) + lambda_* np.sum(w**2 , axis=0)

def compute_gradient(y, tx, w):
    """
    computes the gradient using vector computation

    :param y: 
    :type y: numpy array    
    :param tx: extended feature matrix
    :type tx: numpy 2D array
    :param w: weight parameters 
    :type w: numpy array
    :return: gradient   
    :rtype: numpy array
    """
    # convert row arrays to matrices in correct shape
    w = w.reshape((-1,1))
    y = y.reshape((-1,1))

    # calculate e
    e =  y- np.dot(tx,w)
    # calculate gradient
    gradient = np.dot(tx.T,e)
    gradient *= -1/ y.shape[0]
    return np.array(gradient).flatten()
import numpy as np

def compute_loss(y, tx, w):
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
    w = np.matrix(w).T
    y = np.matrix(y).T
    
    # calculate e
    e =  y- np.dot(tx,w)
    
    #calculate loss
    loss =  np.dot(e.T,e)[0,0]/ 2 / y.shape[0] # MSE
    
    return loss
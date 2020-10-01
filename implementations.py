import numpy as np

from helpers.implementation_helpers import compute_mse, compute_gradient 

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ performs linear regression using Gradient Descent

    :param y: labels
    :type y: numpy array
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each column a feature
    :type tx: numpy 2D array
    :param initial_w: initial values for the regression weight vector
    :type initial_w: numpy array
    :param max_iters: number of max iterations for Gradient Descent
    :type max_iters: int
    :param gamma: Gradient Descent step size
    :type gamma: float

    :return: a tuple containing (weight vector, loss value)
    :rtype: (numpy array, float)
    """
    return least_squares_SGD(y,tx,initial_w,max_iters,gamma,batch_size=len(y))

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    """ performs linear regression using Stochastic Gradient Descent

    :param y: labels
    :type y: numpy array
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each column a feature
    :type tx: numpy 2D array
    :param initial_w: initial values for the regression weight vector
    :type initial_w: numpy array
    :param max_iters: number of max iterations for Gradient Descent
    :type max_iters: int
    :param gamma: Gradient Descent step size
    :type gamma: float
    :param batch_size: [description], defaults to 1
    :type batch_size: int, optional

    :return: a tuple containing (weight vector, loss value)
    :rtype: (numpy array, float)
    """

    n = len(y)
    w = initial_w
    for iter in range(max_iters):
        #create batch
        batch_indices = np.random.randint(n,size=batch_size)
        batch_y = y[batch_indices]
        batch_tx = tx[batch_indices]
        #compute gradient on batch
        gradient = compute_gradient(batch_y,batch_tx,w)
        #print(f"gradient = {gradient}")
        # update w
        w = w - gamma*gradient
    
    loss = compute_mse(y,tx,w)
    return (w,loss)
        

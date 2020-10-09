import numpy as np

from helpers.implementation_helpers import compute_mse, compute_gradient, compute_ridge_loss, calculate_sigmoid, hypothesis_gradient, compute_sigmoid_gradient, compute_loss

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
        
    
def normal_least_squares(y, tx):
    """
    :param y: labels
    :type y: numpy array
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :return: weight, loss value
    :rtype:  numpy array,  float64
    
    """
    A = (tx.T).dot(tx)
    b = (tx.T).dot(y)
    
    #We solve the linear equation Aw = b
    w = np.linalg.solve(A,b)
    
    mse = compute_mse(y, tx, w)
    
    return w , mse

    
def ridge_regression(y, tx, lambda_):
    """
    :param y: labels
    :type y: numpy array
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each column a feature
    :type tx: numpy 2D array
    :param lambda_: Regularizer
    :type lambda_: float64
   
    :return: ridge_weight, loss value
    :rtype:  numpy array,  float64
    """     
    N = len(y)
    lambda_prime = lambda_*(2*N)
    
    A = tx.T.dot(tx) + lambda_prime*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    
    #We solve the linear equation Aw = b
    w_ridge = np.linalg.solve(A,b)
    
    ridge_mse  = compute_ridge_loss(y, tx, w_ridge , lambda_)
    
    return w_ridge , ridge_mse
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    #Logistic regression using gradient descent
    
    :param y: labels
    :type y: numpy 1D array
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param initial_w: initial value of weights
    :type initial_w: numpy 1D array
    
    :param max_iters: the number of maximal iterations
    :type max_iters: int
    
    :param gamma: learning rate
    :type gamma: float64
    
    :return: ws (), losses (the value of the loss function at the end of learning)
    :rtype:  numpy array,  float64
    
    """
    
    # number of features
    D = np.shape(tx)[1]
    # Define parameters to store weights and the value of the loss(cost) funcion
    loss = 0.0
    w = initial_w
    for n_iter in range(max_iters):
        # compute the gradient at the given point
        gradient = compute_sigmoid_gradient(y, tx, w)
        # update weights accordint to the BGD
        w = w - gamma * gradient
    # calculate loss function
    loss = compute_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    #Regularized logistic regression using gradient descent
    
    :param y: labels
    :type y: numpy 1D array
    
    :param lambda_: trade-off parameter (how big part of the loss/cost function to give to regularization function)
    :type lambda_: float64
    
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param initial_w: initial value of weights
    :type initial_w: numpy 1D array
    
    :param max_iters: the number of maximal iterations
    :type max_iters: int
    
    :param gamma: learning rate
    :type gamma: float64
    
    :return: ws (), losses (the value of the loss function at the end of learning)
    :rtype:  numpy array,  float64
    
    """
    
    # number of features
    D = np.shape(tx)[1]
    # Define parameters to store weights and the value of the loss(cost) funcion
    loss = 0.0
    w = np.zeros(D)
    for n_iter in range(max_iters):
        # compute the gradient at the given point
        gradient = compute_sigmoid_gradient(y, tx, w, lambda_)
        # update weights accordint to the BGD
        w = w - gamma * gradient
    # calculate loss function
    loss = compute_loss(y, tx, w, lambda_)
    return w, loss

def reg_batch_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size):
    """
    #Regularized logistic regression using gradient descent
    
    :param y: labels
    :type y: numpy 1D array
    
    :param lambda_: trade-off parameter (how big part of the loss/cost function to give to regularization function)
    :type lambda_: float64
    
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param initial_w: initial value of weights
    :type initial_w: numpy 1D array
    
    :param max_iters: the number of maximal iterations
    :type max_iters: int
    
    :param gamma: learning rate
    :type gamma: float64
    
    :return: ws (), losses (the value of the loss function at the end of learning)
    :rtype:  numpy array,  float64
    
    """
    n = len(y)
    
    # number of features
    D = np.shape(tx)[1]
    # Define parameters to store weights and the value of the loss(cost) funcion
    loss = 0.0
    w = np.zeros(D)
    for n_iter in range(max_iters):
        batch_indices = np.random.randint(n,size=batch_size)
        batch_y = y[batch_indices]
        batch_tx = tx[batch_indices]
        # compute the gradient at the given point
        gradient = compute_sigmoid_gradient(batch_y, batch_tx, w, lambda_)
        # update weights accordint to the BGD
        w = w - gamma * gradient
    # calculate loss function
    loss = compute_loss(y, tx, w, lambda_)
    return w, loss

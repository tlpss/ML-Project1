import numpy as np

from helpers.implementation_helpers import compute_mse, compute_gradient, compute_ridge_loss

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

def calculate_sigmoid(w,tx_ith_point):
    """
    Function calculates the value of sigmoid function at the dot product between w and tx
    
    :param w: extended weights from linear regression
    :type w: numpy 1D array
    
    :param tx: i-th point of extended feature matrix
    :type tx: numpy 1D array
    
    :return: value of sigmoid function for that particular w and tx
    :rtype:  float64
    
    """
    #inner product of two vectors
    inner_product = np.dot(w,tx_ith_point)
    # sigmoid function
    return 1/(1 + np.exp(-inner_product))

def hypothesis_gradient(tx,h):
    """
    Function calculates the value of the gradient of the sigmoid function for every
    i-th point of tx ( i in {0,...,(N-1)} )
    
    Function return a matrix (Jacobian), because we are taking the gradient of a vector
    of sigmoid hypothesis for all points
    
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param h: value of the sigmoid (hypothesis) for the point tx[i] and for the weights w
        sigmoid_x_i = calculate_sigmoid(w, tx[i]) 
    :type h: numpy 1D array
    
    :return: the gradient at the for every i-th hypothesis ( i in {0,...,(N-1)} )
    Functions returns the matrix same dimensions as tx matrix, but every i-th row of tx
    is multiplied by a sigmoid value of i-th hypothesis h[i]
    :rtype:  float64
    
    """
    N,D = np.shape(tx)
    gradient = np.ones([N,D])
    for i in range(N):
        # set the ith row to the gradient value
        # gradient value = 
        # the derivative of the (1D nparray) sigmoid-hypothesis over the (1D nparray) weights w
        gradient[i] = h[i] * (1-h[i]) * tx[i] 
    return gradient
    

def compute_gradient(y, tx, w, lambda_ = 0):
    """
    Function calculates the value of the gradient at the point tx
    
    :param y: labels
    :type y: numpy 1D array
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param w: extended weights from linear regression
    :type w: numpy 1D array
    
    :return: the gradient at the particular point tx and label y
    :rtype:  float64
    
    """
    N,D = np.shape(tx)
    # linear sigmoid (logistic) hypothesis for every point
    h = np.array([calculate_sigmoid(w,tx[i]) for i in range(N)])
    # error vector
    # how much hypothesis h is different from the given label vector y
    e = y - h
    # total number of labels
    N = len(y)
    gradient = - np.matmul(np.transpose(hypothesis_gradient(tx,h)),e)/N + 2 * lambda_ * w
    return gradient

def compute_loss(y, tx, w, lambda_ = 0, MSE = True):
    """
    Function calculates the value of the loss function at the point tx, how much the hypothesis differes form y label vector
    
    :param y: labels
    :type y: numpy 1D array
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param w: extended weights from linear regression
    :type w: numpy 1D array
    
    :return: the error (the residual) between the model h (hypothesis) and the real value
    :rtype:  float64
    
    """
    N,D = np.shape(tx)
    # linear sigmoid (logistic) hypothesis for every point
    h = np.array([calculate_sigmoid(w,tx[i]) for i in range(N)])
    # error vector -- how much hypothesis h is different from the given label vector y
    e = y - h
    if (MSE):
        #MSE
        return np.dot(e,e)/ (2*N) + lambda_ * np.dot(w,w)
    else:
        #MAE
        return np.sum(np.abs(e))/ (2*N) + lambda_ * np.sum(np.abs(w))
    
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
    w = np.zeros(D)
    for n_iter in range(max_iters):
        # compute the gradient at the given point
        gradient = compute_gradient(y, tx, w)
        # update weights accordint to the BGD
        w = w - gamma * gradient
    # calculate loss function
    loss = compute_loss(y, tx, w)
    return loss, w

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
        gradient = compute_gradient(y, tx, w, lambda_)
        # update weights accordint to the BGD
        w = w - gamma * gradient
    # calculate loss function
    loss = compute_loss(y, tx, w, lambda_)
    return loss, w
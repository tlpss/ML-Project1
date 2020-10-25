import numpy as np

'''
LINEAR REGRESSION HELPERS
'''

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


'''
LOGISTIC REGRESSION HELPERS
'''

def sigmoid(x):
    """
    Function calculates the value of sigmoid function on the data structure x
    
    :param x: usulally this is X^T * w
    :type x: scalar or numpy 1D array or numpy 2D array
    
    :return: value of sigmoid function for that particular x
    :rtype:  float64
    
    """
    #return 1/(1 + np.exp(-x))
    return .5 * (1 + np.tanh(.5 * x))

def gradient_of_logistic_regression(tx,y,w,lambda_ = 0):
    """
    Function calculates the value of the gradient of the loss function for every
    i-th point of tx ( i in {0,...,(N-1)} )
    
    Function returns a vector, take a look at lecture 05b in Machine Learning:
        file name : lecture5b_logistic_regression.pdf --->    https://github.com/epfml/ML_course/blob/master/lectures/05/lecture5b_logistic_regression.pdf
        page : 10/17
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param y: binary labels of our data
    :type y: numpy 1D array
    
    :param w: weight parameters for linear regression for every data point in tx
    :type w: numpy 1D array
    
    
    :return: the gradient vector for every data point i-th in tx ( i in {0,...,(N-1)} )
    in the logistic regression
    :rtype:  numpy 1D array
    
    """
    lin_reg = np.matmul(tx,w) # linear regression for every data point in tx
    hypothesis_vector = sigmoid(lin_reg) # just sigmoid applied to every entry of linear regression
    # calculate gradient vector with regularization (of course if lambda_ == 0 then it is without regularization)
    gradient_vector = np.matmul(np.transpose(tx), hypothesis_vector-y) + 2 * lambda_ * w
    return gradient_vector

def loss_of_logistic_regression(tx,y,w,lambda_ = 0):
    """
    Function calculates the value of the loss function for linear regression at the point tx
    
    Function returns a number float64, take a look at lecture 05b in Machine Learning:
        file name : lecture5b_logistic_regression.pdf --->    https://github.com/epfml/ML_course/blob/master/lectures/05/lecture5b_logistic_regression.pdf
        page : 07/17
    
    :param y: labels
    :type y: numpy 1D array
    
    :param tx: extended (contains bias column) feature matrix
    :type tx: numpy 2D array
    
    :param w: weights from linear regression
    :type w: numpy 1D array
    
    :return: the loss function calculated for every data point in tx and then summed up
    :rtype:  float64
    
    """
    # linear regression
    lin_reg = np.matmul(tx,w) 
    # log part of loss function
    log_lin_reg = np.logaddexp(0,lin_reg)  #  = log(1+ exp(lin_reg)) but avoids numerical instability issues
    # linear part of loss function
    # where y_i is zero that doesnt have any effect on the loss
    # where y_i is one that we subtract linear regression part
    y_lin_reg = y * lin_reg 
    loss = np.sum(log_lin_reg - y_lin_reg)
    return loss

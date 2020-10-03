import numpy as np
from helpers.implementation import *

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


def RMSE(mse):
    """
    Calculates RMSE from MSE
    :param mse: mean square error function
    :type y: float64
    
    :return: loss value
    :rtype:  float64
    
    """
    return np.sqrt(2*mse)

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







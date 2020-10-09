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
    

def compute_sigmoid_gradient(y, tx, w, lambda_ = 0):
    """
    Function calculates the value of the gradient at the point tx
    
    :param y: labels
    :type y: numpy 1D array
    
    :param tx: extended (contains bias column) feature matrix, where each row is a datapoint and each          column a feature
    :type tx: numpy 2D array
    
    :param w: extended weights from linear regression
    :type w: numpy 1D array
    
    :return: the gradient at the particular point tx and label y
    :rtype:  numpy 1D array
    
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
    
def basic_split_data(x, y, ratio, seed):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    permutated_indices = np.random.permutation(len(y))
    
    test_len = int(len(y)*ratio)
    
    x_train = x[permutated_indices[: test_len]]
    x_test  = x[permutated_indices[test_len :]]
    y_train = y[permutated_indices[: test_len]]
    y_test  = y[permutated_indices[test_len :]]
    
    return x_test , x_train ,y_test , y_train
    
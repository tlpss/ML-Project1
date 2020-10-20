import matplotlib.pyplot as plt
from implementations import *
import numpy as np
def make_tx(x):
    if np.shape(x)[0] == len(x):
        N = np.shape(x)[0]
        D = 1
    else:
        N,D = np.shape(x)
    ones_N = np.ones(N)
    tx = np.c_[ones_N,x]
    return tx
def find_where_function_is_almost_constant(data, epsilon, is_data_sorted = False):
    diff_data = np.diff(data)
    if (is_data_sorted):
        return np.searchsorted(abs(diff_data) <=epsilon)
    indices = np.where(abs(diff_data) <= epsilon)[0]
    if len(indices) == 0:
        return -1
    else:
        return indices[0]

def generate_different_gammas(linear_gamma):
    # set gamma (learning rate) 
    if (linear_gamma):
        # linear gamma
        gamma_0 = 1e-4
        start_gamma = gamma_0
        stop_gamma = 10*gamma_0
        step_gamma = 2*gamma_0
        gammas = np.arange(start_gamma, stop_gamma, step_gamma)
    else:
        # exponential gamma
        start_degree = -4 #1e-7
        stop_degree = -1 #1e-3
        number_of_points = stop_degree - start_degree + 1
        gammas = np.logspace(start_degree, stop_degree, number_of_points)
        
        start_gamma = 10**start_degree
        stop_gamma = 10**stop_degree
        step_gamma = 10**number_of_points
        
    FRACTION_PRECISION = 2
    gamma_fractions = np.array([[1/x, x] for x in range(1, FRACTION_PRECISION + 1)])
    gamma_fractions  = gamma_fractions .reshape(np.size(gamma_fractions))
    
    
    gammas = np.array([[gamma_fractions * gamma] for gamma in gammas]) 
    gammas = gammas.reshape(np.size(gammas)) # put into a huge vector/array
    gammas = np.sort(gammas) # sort it
    
    return gammas
def plot_cost_max_iter(y,x,regression_type,lambda_ = 0):
    # generate gammas
    gammas = generate_different_gammas(linear_gamma = False)
    
    # set maximum number of iterations
    MAXIMUM_NUMBER_OF_ITERATIONS = int(1e4)
    start_max_iters = 1
    stop_max_iters = MAXIMUM_NUMBER_OF_ITERATIONS+1
    step_max_iters = 1
    max_iters = np.arange(start_max_iters, stop_max_iters, step_max_iters)
    
    # add 1s to the first column of x
    tx = make_tx(x) # add ones
    # extract dimensions
    N,D = np.shape(tx)
    # initial vector of values for weights
    initial_w = np.zeros(D) 
    # save value of loss\cost function
    all_losses = []
    # save value of weights for different models
    all_weights = []
    # for each gamma save the location where the loss curve
    # is almost flat, or in other words where the difference of
    # two consecutive points is smaller then EPSILON
    EPSILON = 1e-5
    all_max_iterations_with_epsilon_precision = []
    MINIMUM_NUMBER_OF_ITERATIONS = 1e10
    MINIMUM_NUMBER_OF_ITERATIONS_CURRENT = 1e100
    all_losses = []
    for gamma in gammas:
        # model(hypothesis) is reg. logistic regression
        if (regression_type == "reg_logistic_regression"):
            w,losses = reg_logistic_regression(y, tx, lambda_, initial_w, MAXIMUM_NUMBER_OF_ITERATIONS, gamma, True)
        # model(hypothesis) is logistic regression
        if (regression_type == "logistic_regression"):
            w,losses,MINIMUM_NUMBER_OF_ITERATIONS_CURRENT = logistic_regression_smart(y, tx, initial_w, MAXIMUM_NUMBER_OF_ITERATIONS, gamma, EPSILON, True, True)
        
        # IF LOSS FUNCTION IS INCREASING THEN WE HAVE TO BIG GAMMA
        # and because gamma is increasing here, that means it will be even
        # bigger and bigger; therefore, we terminate for loop
        # and plot all previous loss functions for different gammmas
        
        if (not np.all(np.diff(losses)<=0)):
            gammas = gammas[:(np.where(gammas == gamma)[0][0] - 1)]
            break
        
        # remember only the smallest amount of
        # maximum number of iterations
        MINIMUM_NUMBER_OF_ITERATIONS = min(MINIMUM_NUMBER_OF_ITERATIONS_CURRENT, MINIMUM_NUMBER_OF_ITERATIONS)
        
        # save the value of the loss function for fixed gamma and max_iter
        all_losses.append([losses])
        
        # save weights when the loss function is the smalles, or in other words
        # save weights with the biggest number max_iter
        all_weights.append(np.array(w))
        
    for loss in all_losses:
        # plot maximum number of iterations (x-axis) and cost/
        curr_loss = loss[0]
        curr_loss = np.array(curr_loss[:MINIMUM_NUMBER_OF_ITERATIONS])
        curr_max_iters = max_iters[:MINIMUM_NUMBER_OF_ITERATIONS]
        plt.plot(curr_max_iters,curr_loss)
        
        
    plt.ylabel('Value of cost/loss function')
    plt.xlabel('Max. number of iterations')
    plt.title('Cost functions for different gammas over of max. number of iterations (epsilon = ' + str(EPSILON) + ')')
    plt.grid(True, 'both')
    
    #plt.legend(handles=[p1, p2], title='title', , loc='upper left', prop=fontP)
    #plt.legend([p1, p2], , bbox_to_anchor=(1.05, 1), , prop=fontP)
    
    #LEGEND
    LEGEND_TAG = [['gamma_ ' + str(i) +' = '+str(gammas[i])] for i in range(len(gammas))] 
    plt.legend(LEGEND_TAG,title='LEGEND\n Gamma_i //  Constant after max_inter_i\n (-1 means function is not const for given epsilon)',bbox_to_anchor=(1.8, 1),loc='upper right')
    
    plt.xlim(start_max_iters, MINIMUM_NUMBER_OF_ITERATIONS)     # set the xlim to left, right
    # plt.yscale("log")
    plt.show()
    print("GAMMAS: ",gammas)
    print("The function is constant (two points are less than epsilon = "+str(EPSILON)+" ) after each max_iteration (for each gamma, respectively): \n", all_max_iterations_with_epsilon_precision, "\n  ~(-1 means that function is never constant for a given plot)~ ")
    #return np.array(all_weights), gammas
    return np.array(all_weights), np.array(all_losses), gammas, all_max_iterations_with_epsilon_precision


# data points
x = np.array(np.arange(1,21,1))
#labels
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1])
# plot cost function over max. number of iterations for each gamma
all_weights, all_losses, gammas, all_max_iterations_with_epsilon_precision = plot_cost_max_iter(y,x,"logistic_regression", 0)
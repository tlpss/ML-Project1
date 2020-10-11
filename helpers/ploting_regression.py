import matplotlib.pyplot as plt
from logistic_regression import *
def make_tx(x):
    N,D = np.shape(x)
    ones_N = np.ones(N)
    tx = np.c_[ones_N,x]
    return tx
def plot_cost_max_iter(y,x,regression_type,lambda_ = 0):
    # set gamma (learning rate)
    start_gamma = 0.1 
    stop_gamma = 1.8
    step_gamma = 0.3
    gammas = np.arange(start_gamma, stop_gamma, step_gamma)
    # set maximum number of iterations
    start_max_iters = 1
    stop_max_iters = 101
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
    
    for gamma in gammas:
        all_losses = []
        for max_iter in max_iters:
            # model(hypothesis) is reg. logistic regression
            if (regression_type == reg_logistic_regression):
                loss,w = regression_type(y, tx, lambda_, initial_w, max_iter, gamma)
            # model(hypothesis) is logistic regression
            if (regression_type == logistic_regression):
                loss,w = regression_type(y, tx, initial_w, max_iter, gamma)
            # save the value of the loss function for fixed gamma and max_iter
            all_losses.append(loss)
        # save weights when the loss function is the smalles, or in other words
        # save weights with the biggest number max_iter
        all_weights.append(np.array(w))
        # plot maximum number of iterations (x-axis) and cost/
        plt.plot(max_iters,all_losses)
    plt.ylabel('Value of cost/loss function')
    plt.xlabel('Max. number of iterations')
    plt.title('Cost func for '+str(start_gamma)+' <= gamma < '+str(stop_gamma)+ ' in function of max. number of iterations')
    plt.grid(True, 'both')
    plt.legend([['gamma='+str(np.round(gamma,2))] for gamma in gammas])
    plt.xlim(start_max_iters, stop_max_iters)     # set the xlim to left, right
    plt.show()
    return np.array(all_weights), (start_gamma, stop_gamma, step_gamma)

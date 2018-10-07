import numpy as np
import matplotlib.pyplot as plt
import math

def logistic_1d(x, v, b):
    return 1.0 / (1 + np.exp([-v*x[i]-b for i in range(len(x))]))

def rbf_1d(x, c = 0, h = 1):
    return np.exp([-(x[i]-c)**2/ h**2 for i in range(len(x))])

def linear_transform_1d(X):
    no_bias_linear = np.reshape(np.array(X), (-1, 1))
    #print(no_bias_linear.shape)
    return np.concatenate((no_bias_linear, np.ones((no_bias_linear.shape[0], 1))), axis = 1)

def rbf_transform_1d(X):
    rbf_c1 = np.array(rbf_1d(X, 1, 1)).reshape((-1, 1))
    rbf_c2 = np.array(rbf_1d(X, 2, 1)).reshape((-1, 1))
    rbf_c3 = np.array(rbf_1d(X, 3, 1)).reshape((-1, 1))
    return np.concatenate((rbf_c1, rbf_c2, rbf_c3), axis = 1)

def polynomial_transform_1d(X, order):
    #quad_1 = np.array(X).reshape(-1, 1)
    #uad_2 = np.array(np.square(X)).reshape(-1, 1)
    #quad_3 = np.array(np.power(X, 3)).reshape(-1, 1)
    quad_n = [np.array(np.power(X, i)).reshape(-1, 1) for i in range(1, order + 1)]
    quad_n.append(np.ones((len(X), 1)))
    return np.concatenate(tuple(quad_n), axis = 1)

def fit_plot(X, input, yy, transform_function, grid_size = 0.01, order = None):
    yy = np.reshape(np.array(yy), (-1, 1))
    W = np.linalg.lstsq(X, yy, rcond = None)[0]
    #yy_pred = np.dot(X, W).tolist()

    x_grid = np.arange(-2, 2, grid_size)
    y_grid = np.dot(transform_function(x_grid, order = order), W).tolist() if order != None else np.dot(transform_function(x_grid), W).tolist()
    #plt.clf()
    plt.plot(x_grid, y_grid, 'b-')
    plt.plot(input, yy, 'r.')
    #plt.show()
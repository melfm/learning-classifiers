#!/usr/bin/env python2
import processData
import numpy as np
import matplotlib.pyplot as plt


def linear_regression_compute_cost(theta, X, y):
    '''
    Compute the linear regression objective function
    Cost function : J(theta) = 1/2m * sum (h_theta(x_i) - y_i)^2
    Arguments:
    theta - A vector containing the parameter values to optimize
    X - The training examples which is a vector
    y - The target values for each example
    '''
    m = y.size
    # Convert to 1D vector
    prediction = X.dot(theta).flatten()
    S = (prediction - y) ** 2
    J = (1.0/(2*m)) * S.sum()
    return J


def step_gradient(theta, X, y, alpha, num_iters):
    '''
    Calculate the gradient by differentiating the error function.
    '''
    m = y.size
    J_costs = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        predictions = X.dot(theta).flatten()
        error_x1 = (predictions - y) * X[:, 0]
        error_x2 = (predictions - y) * X[:, 1]
        theta[0][0] = theta[0][0] - alpha * (1.0/m) * error_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0/m) * error_x2.sum()
        J_costs[i, 0] = linear_regression_compute_cost(theta, X, y)
    return theta, J_costs


def run_gradient_descent(X, y, alpha, num_iterations):
    '''
    Performs gradient descent to learn theta
    taking num_iters gradient steps
    '''
    # Initialize theta to some small random values
    theta = np.random.random_sample((2, 1)) * 0.001
    theta, J_costs = step_gradient(
        theta,
        x_input,
        y,
        alpha,
        num_iterations
    )

    print theta
    return theta, J_costs


def plot_results(data, x_input, theta):
    result = x_input.dot(theta).flatten()
    plt.figure(1)
    plt.plot(data[:, 0], result)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show(block=False)

    # Grid over which we will calculate J
    theta_0_vals = np.linspace(-10, 10, 100)
    theta_1_vals = np.linspace(-1, 4, 100)

    # Initialize J_vals to a matrix of 0's
    J_vals = np.zeros(shape=(theta_0_vals.size, theta_1_vals.size))

    # Fill out the J_vals
    for t1, element in enumerate(theta_0_vals):
        for t2, element2 in enumerate(theta_1_vals):
            thetaT = np.zeros(shape=(2, 1))
            thetaT[0][0] = element
            thetaT[1][0] = element2
            J_vals[t1, t2] = linear_regression_compute_cost(thetaT, x_input, y)

    # Contour plot
    J_vals = J_vals.T
    plt.figure(2)
    plt.contour(theta_0_vals, theta_1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.scatter(theta[0][0], theta[1][0])
    plt.show()

if __name__ == "__main__":
    data = processData.LinearRegData(False)
    X = data[:, 0]
    y = data[:, 1]

    m = y.size
    # Include a row of 1s as an additional intercept feature
    x_input = np.ones(shape=(m, 2))
    x_input[:, 1] = X

    alpha = 0.01
    num_iterations = 1500
    theta, J_costs = run_gradient_descent(x_input, y, alpha, num_iterations)
    plot_results(data, x_input, theta)

import numpy as np
def hypothesis(X, theta):
    # h_theta(x) = theta^T x for every row at once
    return X @ theta

def cost(X, y, theta):
    # J(theta) = 1/2 * sum of squared errors (no 1/n, as in Lecture 1)
    errors = hypothesis(X, theta) - y
    return 0.5 * np.sum(errors ** 2)

def fit_normal(X, y):
    # theta = (X^T X)^-1 X^T y
    return np.linalg.inv(X.T @ X) @ X.T @ y

def fit_batch_gd(X, y, alpha, n_iters):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(n_iters):
        errors = y - hypothesis(X, theta)

        # theta_j := theta_j + alpha * sum_i (y_i - h(x_i)) * x_ij
        theta = theta + alpha * (X.T @ errors)

        cost_history.append(cost(X, y, theta))

    return theta, cost_history

def fit_sgd(X, y, alpha, n_epochs):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(n_epochs):
        # One update per training row, in time order
        for i in range(len(y)):
            error = y[i] - X[i] @ theta
            theta = theta + alpha * error * X[i]

        cost_history.append(cost(X, y, theta))

    return theta, cost_history


def rmse(y, predictions):
    return np.sqrt(np.mean((y - predictions) ** 2))

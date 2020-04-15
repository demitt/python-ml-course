import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, W, B):
    return sigmoid(np.dot(W, X) + B)


def cost(X, W, B, Y):
    hypothesis_tr = h(X, W, B).T
    m = X.shape[1]
    value = (- np.dot(Y, np.log(hypothesis_tr + 10E-10)) - np.dot((1 - Y), np.log(1 - hypothesis_tr + 10E-10))) / m
    return value[0, 0]


def cost_gradients(X, W, B, Y):
    delta = h(X, W, B) - Y
    m = X.shape[1]

    return dict(
        W=np.dot(delta, X.T) / m,
        B=np.sum(delta) / m
    )


def minimize_cost(X, Y, alpha, iterations_count):
    n = X.shape[0]

    W = np.zeros((1, n))
    B = np.zeros((1, 1))

    for iteration in range(iterations_count):
        gradients = cost_gradients(X, W, B, Y)
        W = W - alpha * gradients['W']
        B = B - alpha * gradients['B']
        # print(cost(X, W, B, Y))

    return dict(W=W, B=B)

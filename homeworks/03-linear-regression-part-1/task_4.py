import numpy as np
from TaskVerifier import TaskVerifier
from task_1 import h
from task_2 import cost
from task_3 import cost_gradients

# Нужно реализовать итерацию метода градиентного спуска.
# alpha -- шаг алгоритма, N_iter -- количество итераций.
# X: (n, m)
# Y: (1, m)
# W: (1, n)
# b: (1, 1)
def minimize_cost(X, Y, alpha, N_iter):
    n = X.shape[0]

    W = np.zeros((1, n))
    b = np.zeros((1, 1))

    for iteration in range(N_iter):
        # НАЧАЛО ЗАДАНИЯ

        gradients = cost_gradients(X, Y, W, b)
        W_new = W - alpha * gradients['W']
        b_new = b - alpha * gradients['b']

        W = W_new
        b = b_new

        # КОНЕЦ ЗАДАНИЯ

    return dict(W=W, b=b)


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_4',
        'minimize_cost',
        [
            np.array([[2], [3], [-4.5]]),
            np.array([[1.3]]),
            0.01,
            5000
        ],
        dict(W=np.array([[ 0.07591241, 0.11386861, -0.17080292]]), b=np.array([[0.0379562]]))
    )

    task_verifier.test(
        'task_4',
        'minimize_cost',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[2.1, -5.7]]),
            0.001,
            10000
        ],
        dict(W=np.array([[-1.25757491, -0.21295499, -0.85608622]]), b=np.array([[-0.2777751]]))
    )
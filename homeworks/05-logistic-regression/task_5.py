import numpy as np
from TaskVerifier import TaskVerifier
from task_2 import h
from task_3 import cost
from task_4 import cost_gradients


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

        # КОНЕЦ ЗАДАНИЯ

    return dict(W=W, b=b)


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_5',
        'minimize_cost',
        [
            np.array([[2], [3], [-4.5]]),
            np.array([[1]]),
            0.01,
            5000
        ],
        dict(W=np.array([[0.43460563, 0.65190844, -0.97786266]]), b=np.array([[0.21730281]]))
    )

    task_verifier.test(
        'task_5',
        'minimize_cost',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[0, 1]]),
            0.001,
            10000
        ],
        dict(W=np.array([[0.90164845, -0.00473734, 1.14051722]]), b=np.array([[0.14899315]]))
    )

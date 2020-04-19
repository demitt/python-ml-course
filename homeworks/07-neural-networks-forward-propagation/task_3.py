import numpy as np
from TaskVerifier import TaskVerifier
from task_2 import forward_propagation


# Функция должна вернуть значение cost function, вычисленной на данных X и Y с параметрами W1, b1, W2, b2.
# X: (n, m)
# W1: (n1, n)
# b1: (n1, 1)
# W2: (1, n1)
# b2: (1, 1)
def cost(X, Y, W1, b1, W2, b2):
    m = X.shape[1]

    value = 0

    # НАЧАЛО ЗАДАНИЯ
    # ...

    # КОНЕЦ ЗАДАНИЯ

    return value


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_3',
        'cost',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[0, 1]]),
            np.array([[1.0, -1.5, 2.5], [2.4, -1.2, 2.2], [-2.3, 1.1, -0.1]]),  # (3, 3)
            np.array([[0.3], [1.3], [-0.2]]),  # (3, 1)
            np.array([[0.2, -3.8, 4.6]]),  # (1, 3)
            np.array([[1.5]])  # (1, 1)
        ],
        2.6071230221099113
    )
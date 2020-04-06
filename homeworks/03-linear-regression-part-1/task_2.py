import numpy as np
from TaskVerifier import TaskVerifier
from task_1 import h


# Функция должна вернуть значение cost function, вычисленной на данных X и Y с параметрами W и b.
# X: (n, m)
# Y: (1, m)
# W: (1, n)
# b: (1, 1)
# return: число
def cost(X, Y, W, b):
    m = X.shape[1]

    value = 0

    # НАЧАЛО ЗАДАНИЯ

    delta = h(X, W, b) - Y
    value_as_matrix = 0.5 / m * np.dot(delta, delta.T)
    value = value_as_matrix[0, 0]

    # КОНЕЦ ЗАДАНИЯ

    return value


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_2',
        'cost',
        [
            np.array([[2], [3], [-4.5]]),
            np.array([[1.3]]),
            np.array([[1.0, -1.5, 2.5]]),
            np.array([[3.5]])
        ],
        66.70125
    )

    task_verifier.test(
        'task_2',
        'cost',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[2.1, -5.7]]),
            np.array([[2.3, -1.5, 2.5]]),
            np.array([[1.8]])
        ],
        112.966625
    )
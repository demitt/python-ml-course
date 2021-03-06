import numpy as np
from TaskVerifier import TaskVerifier

# Функция должна вернуть значение гипотезы h, вычисленной на данных X с параметрами W и b
# X: (n, m)
# W: (1, n)
# b: (1, 1)
# return: (1, m)
def h(X, W, b):
    value = 0

    # НАЧАЛО ЗАДАНИЯ

    value = np.dot(W, X) + b

    # КОНЕЦ ЗАДАНИЯ

    return value


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_1',
        'h',
        [
            np.array([[2], [3], [-4.5]]),
            np.array([[1.0, -1.5, 2.5]]),
            np.array([[3.5]])
        ],
        [[-10.25]]
    )

    task_verifier.test(
        'task_1',
        'h',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[2.3, -1.5, 2.5]]),
            np.array([[1.8]])
        ],
        np.array([[-12.22, 10.01]])
    )

import numpy as np
from TaskVerifier import TaskVerifier
from task_1 import sigmoid


# Предполагается, что нейронная сеть имеет один скрытый слой из n1 нейронов.
# Функция должна вернуть словарь с ключами l1 и l2.
# result['l1'] - матрица активаций в скрытом слое (матрица размерностью (n1, m)).
# result['l2'] -- матрица активаций в выходном слое, фактически гипотеза (матрица размерностью (1, m)).
#
# Параметры:
# X: (n, m)
# W1: (n1, n)
# b1: (n1, 1)
# W2: (1, n1)
# b2: (1, 1)
def forward_propagation(X, W1, b1, W2, b2):
    result = dict(l1=None, l2=None)

    # НАЧАЛО ЗАДАНИЯ

    a1 = sigmoid(np.dot(W1, X) + b1)
    result = dict(
        l1=a1,
        l2=sigmoid(np.dot(W2, a1) + b2)
    )

    # КОНЕЦ ЗАДАНИЯ

    return result


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    # X: (n, m)
    # W1: (n1, n)
    # b1: (n1, 1)
    # W2: (1, n1)
    # b2: (1, 1)
    task_verifier.test(
        'task_2',
        'forward_propagation',
        [
            np.array([[2], [3], [-4.5]]),  # (3, 1)
            np.array([[1.0, -1.5, 2.5], [2.4, -1.2, 2.2], [-2.3, 1.1, -0.1]]),  # (3, 3)
            np.array([[0.3], [1.3], [-0.2]]),  # (3, 1)
            np.array([[0.2, -3.8, 4.6]]),  # (1, 3)
            np.array([[3.5]])  # (1, 1)
        ],
        {'l1': np.array([[1.44124758e-06], [6.10879359e-04], [2.59225101e-01]]), 'l2': np.array([[0.99089797]])}
    )

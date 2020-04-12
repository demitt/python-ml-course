import numpy as np
from TaskVerifier import TaskVerifier
from task_2 import h
from task_3 import cost


# Функция должна вернуть производные cost function по W и по b.
# Нужно присвоить корректные значения переменным grads_W и grad_b.
# X: (n, m)
# Y: (1, m)
# W: (1, n)
# b: (1, 1)
def cost_gradients(X, Y, W, b):
    m = X.shape[1]

    grads_W = None
    grad_b = None

    # НАЧАЛО ЗАДАНИЯ

    delta = h(X, W, b) - Y
    grads_W = np.dot(delta, X.T) / m
    grad_b = np.sum(delta) / m

    # КОНЕЦ ЗАДАНИЯ

    return dict(W=grads_W, b=grad_b)


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_4',
        'cost_gradients',
        [
            np.array([[2], [3], [-4.5]]),
            np.array([[1]]),
            np.array([[1.0, -1.5, 2.5]]),
            np.array([[3.5]])
        ],
        dict(W=np.array([[-1.99992929, -2.99989393, 4.4998409]]), b=-0.9999646437492583)
    )

    task_verifier.test(
        'task_4',
        'cost_gradients',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[0, 1]]),
            np.array([[2.3, -1.5, 2.5]]),
            np.array([[1.8]])
        ],
        dict(W=np.array([[-7.92057622e-05, -2.12972588e-05, -2.75366934e-05]]), b=-2.0007675532693166e-05)
    )

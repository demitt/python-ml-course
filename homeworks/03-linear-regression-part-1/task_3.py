import numpy as np
from TaskVerifier import TaskVerifier
from task_1 import h
from task_2 import cost

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

    grad_b_as_scalar = np.sum(delta) / m
    grad_b = np.array([[grad_b_as_scalar]])

    # КОНЕЦ ЗАДАНИЯ

    return dict(W=grads_W, b=grad_b)


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_3',
        'cost_gradients',
        [
            np.array([[2], [3], [-4.5]]),
            np.array([[1.3]]),
            np.array([[1.0, -1.5, 2.5]]),
            np.array([[3.5]])
        ],
        dict(W=np.array([[-23.1, -34.65, 51.975]]), b=np.array([[-11.55]]))
    )

    task_verifier.test(
        'task_3',
        'cost_gradients',
        [
            np.array([[1.6, 3.7], [2.3, 1.2], [-5.7, 0.6]]),
            np.array([[2.1, -5.7]]),
            np.array([[2.3, -1.5, 2.5]]),
            np.array([[1.8]])
        ],
        dict(W=np.array([[17.6075, -7.042, 45.525]]), b=np.array([[0.695]]))
    )
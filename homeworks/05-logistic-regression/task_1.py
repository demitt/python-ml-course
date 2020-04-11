import numpy as np
from TaskVerifier import TaskVerifier

# Функция должна присвоить переменой value значение функции sigmoid, вычисленной в точке z.
# Примечание: для экспоненты использовать np.exp().
def sigmoid(z):
    value = 0

    # НАЧАЛО ЗАДАНИЯ
    # ...

    # КОНЕЦ ЗАДАНИЯ

    return value


if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_1',
        'sigmoid',
        [
            0.5
        ],
        0.6224593312018546
    )

    task_verifier.test(
        'task_1',
        'sigmoid',
        [
            np.array(([0.5, -0.2], [-0.4, 1.3]))
        ],
        np.array([[0.62245933, 0.450166], [0.40131234, 0.78583498]])
    )

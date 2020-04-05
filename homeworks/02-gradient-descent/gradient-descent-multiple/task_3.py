import numpy as np
from TaskVerifier import TaskVerifier

# Импортируем то, что уже реализовали ранее
from task_1 import f
from task_2 import df


def minimize_function():
    result = {'x': None, 'f': None}

    # Шаг алгоритма alpha
    alpha = 0.1

    # Стартуем поиск со значения x1 = 10, x2 = 15
    x = np.array([[10, 15]])

    # Общее количество итераций = 1000
    num_iterations = 1000

    # НАЧАЛО ЗАДАНИЯ

    for i in range(num_iterations):
        x = x - alpha * df(x[0, 0], x[0, 1])

    result = {
        'x1': x[0, 0],
        'x2': x[0, 1],
        'f': f(x[0, 0], x[0, 1])
    }

    # КОНЕЦ ЗАДАНИЯ

    return result


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_3',
        'minimize_function',
        [
        ],
        {'x1': 2., 'x2': -2.5, 'f': -13.25}
    )

import numpy as np
from TaskVerifier import TaskVerifier


# Функция ниже должна вернуть матрицу 3x5, состоящую из нулей.
# Используйте функции модуля numpy.
def generate_zero_matrix():
    matrix = None

    # В строках ниже нужно присвоить переменной matrix нужное значение

    # НАЧАЛО ЗАДАНИЯ

    # КОНЕЦ ЗАДАНИЯ

    return matrix


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_2',
        'generate_zero_matrix',
        [
        ],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )

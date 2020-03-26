import numpy as np
from TaskVerifier import TaskVerifier


# Функция ниже принимает матрицу A в качестве аргумента.
# Она должна вернуть словарь: {'rows': <количество строк в матрице>, 'cols': <количество столбцов в матрице>}.
# Например, если на вход передана матрица размером 2x3, то функция должна вернуть {'rows': 2, 'cols': 3}
def calculate_matrix_shape(A):
    matrix_shape = dict()

    # В строках ниже нужно корректно сформировать словарь matrix_shape

    # НАЧАЛО ЗАДАНИЯ

    # КОНЕЦ ЗАДАНИЯ

    return matrix_shape


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
# В этом примере запускается не один, а два теста.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_3',
        'calculate_matrix_shape',
        [
            np.array([[1, 2, 3], [4, 5, 6]])
        ],
        dict(rows=2, cols=3)
    )

    task_verifier.test(
        'task_3',
        'calculate_matrix_shape',
        [
            np.array([[1], [2], [3], [4], [5]])
        ],
        dict(rows=5, cols=1)
    )

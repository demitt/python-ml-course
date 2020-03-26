import numpy as np
from TaskVerifier import TaskVerifier


# Функция ниже принимает матрицу A в качестве аргумента.
# Она должна вернуть транспонированную матрицу A.
def transpose_matrix(A):
    A_transposed = None

    # В строках ниже нужно присвоить корректное значение переменной A_transposed

    # НАЧАЛО ЗАДАНИЯ
    A_transposed = A.T
    # КОНЕЦ ЗАДАНИЯ

    return A_transposed


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
# В этом примере запускается не один, а два теста.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_4',
        'transpose_matrix',
        [
            np.array([[1, 2, 3], [4, 5, 6]])
        ],
        np.array([[1, 4], [2, 5], [3, 6]])
    )

    task_verifier.test(
        'task_4',
        'transpose_matrix',
        [
            np.array([[1], [2], [3], [4], [5]])
        ],
        np.array([[1, 2, 3, 4, 5]])
    )

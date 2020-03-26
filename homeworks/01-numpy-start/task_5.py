import numpy as np
from TaskVerifier import TaskVerifier


# Функция ниже принимает матрицы A и B в качестве аргументов.
# Она должна вернуть результат их _матричного_ умножения A * B.
# При этом, если количество столбцов в матрице A не равно количеству строк в матрице B,
# она должна вернуть строку 'Invalid matrix shape!'
def multiply_matrices(A, B):
    result = None

    # В строках ниже нужно присвоить корректное значение переменной result

    # НАЧАЛО ЗАДАНИЯ
    dimsA = A.shape
    dimsB = B.shape

    if dimsA[1] != dimsB[0]:
        return 'Invalid matrix shape!'

    result = np.dot(A, B)
    # КОНЕЦ ЗАДАНИЯ

    return result


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
# В этом примере запускается не один, а два теста.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_5',
        'multiply_matrices',
        [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[2], [5], [7]])
        ],
        np.array([[33], [75]])
    )

    task_verifier.test(
        'task_5',
        'multiply_matrices',
        [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[2], [5], [7], [9]])
        ],
        'Invalid matrix shape!'
    )

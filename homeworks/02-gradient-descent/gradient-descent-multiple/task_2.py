import numpy as np
from TaskVerifier import TaskVerifier


def df(x1, x2):
    df_x1 = None
    df_x2 = None

    # В строках ниже нужно вычислить значение производные функции x1^2 - 4x1 + 6 + x2^2 + 5x2 - 9 в точке (x1, x2)
    # и присвоить его переменным df_x1, df_x2.
    # Для функции x1^2 - 4x1 + 6 + x2^2 + 5x2 - 9 производные равна 2x1 - 4 и 2x2 + 5, соответственно.

    # НАЧАЛО ЗАДАНИЯ

    df_x1 = 2 * x1 - 4
    df_x2 = 2 * x2 + 5

    # КОНЕЦ ЗАДАНИЯ

    return np.array([[df_x1, df_x2]])


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_2',
        'df',
        [
            0, 0
        ],
        np.array([[-4, 5]])
    )

    task_verifier.test(
        'task_2',
        'df',
        [
            2, 3
        ],
        np.array([[0, 11]])
    )

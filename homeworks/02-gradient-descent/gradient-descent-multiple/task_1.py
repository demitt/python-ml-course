from TaskVerifier import TaskVerifier


def f(x1, x2):
    value = None

    # В строках ниже нужно вычислить значение функции x1^2 - 4x1 + 6 + x2^2 + 5x2 - 9 в точке (x1, x2)
    # и присвоить его переменной value

    # НАЧАЛО ЗАДАНИЯ

    value = x1 ** 2 - 4 * x1 + 6 + x2 ** 2 + 5 * x2 - 9

    # КОНЕЦ ЗАДАНИЯ

    return value


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_1',
        'f',
        [
            0,
            0
        ],
        -3
    )

    task_verifier.test(
        'task_1',
        'f',
        [
            2,
            3
        ],
        17
    )

from TaskVerifier import TaskVerifier


def f(x):
    value = None

    # В строках ниже нужно вычислить значение функции x^2 - 3x + 2 в точке x
    # и присвоить его переменной value

    # НАЧАЛО ЗАДАНИЯ

    value = x ** 2 - 3 * x + 2

    # КОНЕЦ ЗАДАНИЯ

    return value


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_1',
        'f',
        [
            0
        ],
        2
    )

    task_verifier.test(
        'task_1',
        'f',
        [
            2
        ],
        0
    )

from TaskVerifier import TaskVerifier


def df(x):
    value = None

    # В строках ниже нужно вычислить значение производной функции x^2 - 3x + 2 в точке x
    # и присвоить его переменной value.
    # Для функции x^2 - 3x + 2 производная равна 2x - 3

    # НАЧАЛО ЗАДАНИЯ

    # КОНЕЦ ЗАДАНИЯ

    return value


# Здесь не нужно ничего менять. Эти строки запускают автотест, который проверит корректность выполнения вашего задания.
if __name__ == '__main__':
    task_verifier = TaskVerifier()

    task_verifier.test(
        'task_2',
        'df',
        [
            0
        ],
        -3
    )

    task_verifier.test(
        'task_2',
        'df',
        [
            2
        ],
        1
    )

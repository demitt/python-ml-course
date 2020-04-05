from TaskVerifier import TaskVerifier

# Импортируем то, что уже реализовали ранее
from task_1 import f
from task_2 import df


def minimize_function():
    result = {'x': None, 'f': None}

    # Шаг алгоритма alpha
    alpha = 0.1

    # Стартуем поиск со значения x = 10
    x = 10

    # Общее количество итераций = 1000
    num_iterations = 1000

    # НАЧАЛО ЗАДАНИЯ

    for i in range(num_iterations):
        x = x - alpha * df(x)

    result = {
        'x': x,
        'f': f(x)
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
        {'x': 1.5, 'f': -0.25}
    )

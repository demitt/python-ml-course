class TaskVerifier:
    def __init__(self):
        self.task_num = 1

    def test(self, module_name, method_name, params, expected_value):
        module = __import__(module_name)
        method = getattr(module, method_name)
        value = method(*params)

        print('Тест #{}'.format(self.task_num))
        print('Вызов метода {} с параметрами:'.format(method_name))
        print(params)
        print('')
        print('Ожидаемое значение:')
        print(expected_value)
        print('Фактическое значение:')
        print(value)
        print('===========')

        self.task_num += 1

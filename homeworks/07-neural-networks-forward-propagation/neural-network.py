import numpy as np
import matplotlib.pyplot as plt


# X: (n, m)
def normalize_data(X):
    data_min = np.min(X, axis=1, keepdims=True)
    data_max = np.max(X, axis=1, keepdims=True)

    return (X - data_min) / (data_max - data_min)


def read_csv(filename, y_index=-1):
    csv_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    np.random.shuffle(csv_data)
    csv_data = csv_data.T

    m = csv_data.shape[1]
    n = csv_data.shape[0]
    train_count = int(m * 0.7)

    if y_index == -1:
        y_index = n - 1

    data_x = np.delete(csv_data, y_index, 0)
    data_x = normalize_data(data_x)
    data_x_train = data_x[:, 0:train_count]
    data_x_test = data_x[:, train_count:]

    data_y = csv_data[y_index:y_index+1]
    data_y_train = data_y[:, 0:train_count]
    data_y_test = data_y[:, train_count:]

    return dict(X_train=data_x_train, Y_train=data_y_train, X_test=data_x_test, Y_test=data_y_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(a):
    return a * (1 - a)


def relu(z):
    return np.maximum(z, 0)


def d_relu(a):
    return 1 * (a > 0)


def activation_1(z, f='sigmoid'):
    if f == 'sigmoid':
        return sigmoid(z)
    if f == 'relu':
        return relu(z)
    return 0.0


def activation_derivative_1(a, f='sigmoid'):
    if f == 'sigmoid':
        return d_sigmoid(a)
    if f == 'relu':
        return d_relu(a)
    return 0.0


# X: (n, m)
# W['l1']: (L1, n)
# W['l2']: (1, L1)
# b['l1']: (L1, 1)
# b['l2']: (1, 1)
def predict(X, W, b):
    z1 = np.dot(W['l1'], X) + b['l1']
    a1 = activation_1(z1, 'sigmoid')

    z2 = np.dot(W['l2'], a1) + b['l2']
    a2 = sigmoid(z2)

    return dict(a1=a1, a2=a2)


# X: (n, m)
# Y: (1, m)
# W['l1']: (L1, n)
# W['l2']: (1, L1)
# b['l1']: (L1, 1)
# b['l2']: (1, 1)
def cost(X, Y, W, b, reg_lambda=0.0):
    m = X.shape[1]

    cache = predict(X, W, b)
    h = cache['a2']

    value = (-np.dot(Y, np.log(h + 1e-10).T) - np.dot(1 - Y, np.log(1 - h + 1e-10).T))[0, 0] / m
    value += (reg_lambda / (2 * m)) * np.sum(W['l1'] * W['l1'])
    value += (reg_lambda / (2 * m)) * np.sum(W['l2'] * W['l2'])

    return value


# X: (n, m)
# Y: (1, m)
# W['l1']: (L1, n)
# W['l2']: (1, L1)
# b['l1']: (L1, 1)
# b['l2']: (1, 1)
def cost_gradients(X, Y, W, b, cache, m, reg_lambda=0.0):
    a1 = cache['a1']
    a2 = cache['a2']

    temp2 = Y - a2

    grads_W2 = (-1 / m) * np.dot(temp2, a1.T) + (reg_lambda / m) * W['l2']
    grads_b2 = (-1 / m) * np.sum(temp2, axis=1, keepdims=True)

    activation_derivative = activation_derivative_1(a1, 'sigmoid')
    temp1 = np.dot(W['l2'].T, Y - a2) * activation_derivative

    grads_W1 = (-1 / m) * np.dot(temp1, X.T) + (reg_lambda / m) * W['l1']
    grads_b1 = (-1 / m) * np.sum(temp1, axis=1, keepdims=True)

    # cost_value = cost(X, Y, W, b, reg_lambda)
    #
    # W_fixed = W.copy()
    #
    # grads_W1_correct = np.zeros(W['l1'].shape)
    # for i in range(W['l1'].shape[0]):
    #     for j in range(W['l1'].shape[1]):
    #         W1_new = W_fixed['l1'].copy()
    #         W1_new[i, j] += 1e-6
    #         W = dict(l1=W1_new, l2=W_fixed['l2'])
    #         cost_value_new = cost(X, Y, W, b, reg_lambda)
    #         grads_W1_correct[i, j] = (cost_value_new - cost_value) / 1e-6
    #
    # print(grads_W1)
    # print(grads_W1_correct)
    # print('======')

    return dict(W1=grads_W1, b1=grads_b1, W2=grads_W2, b2=grads_b2)


# X: (n, m)
# Y: (1, m)
# W['l1']: (L1, n)
# W['l2']: (1, L1)
# b['l1']: (L1, 1)
# b['l2']: (1, 1)
def minimize_cost(X, Y, reg_lambda=0.0):
    m = X.shape[1]
    n = X.shape[0]

    cost_values = []

    L1 = 2
    W1 = np.random.rand(L1, n)
    W2 = np.random.rand(1, L1)
    b1 = np.random.rand(L1, 1)
    b2 = np.random.rand(1, 1)

    alpha = 0.5

    W = dict(l1=W1, l2=W2)
    b = dict(l1=b1, l2=b2)
    cost_value = cost(X, Y, W, b, reg_lambda)
    print(cost_value)

    for iteration in range(50000):
        W = dict(l1=W1, l2=W2)
        b = dict(l1=b1, l2=b2)

        cache = predict(X, W, b)
        grads = cost_gradients(X, Y, W, b, cache, m, reg_lambda)

        W2 -= alpha * grads['W2']
        W1 -= alpha * grads['W1']
        b1 -= alpha * grads['b1']
        b2 -= alpha * grads['b2']

        cost_value = cost(X, Y, W, b, reg_lambda)
        cost_values.append(cost_value)

        if iteration % 1000 == 0:
            print('{} | {}'.format(iteration, cost_value))

    print('============')
    print(W)
    print(b)
    print('============')

#    plt.plot(cost_values)
#    plt.show()

    return dict(W=W, b=b)


def plot_learning_curves(X_train, Y_train, X_test, Y_test, reg_lambda=0.0):
    m = X_train.shape[1]

    range_x = range(5, m, 40)
    cost_values_train = []
    cost_values_test = []

    for i in range_x:
        X = X_train[:, 0:i]
        Y = Y_train[:, 0:i]

        params_i = minimize_cost(X, Y, reg_lambda)
        cost_values_train.append(cost(X, Y, params_i['W'], params_i['b']))
        cost_values_test.append(cost(X_test, Y_test, params_i['W'], params_i['b']))

    plt.plot(range_x, cost_values_train, range_x, cost_values_test)
    plt.show()


def calculate_precision(X, Y, W, b):
    m = X.shape[1]

    predictions = predict(X, W, b)['a2']

    success_counter = 0

    for i in range(m):
        p = 1.0 if predictions[0, i] >= 0.5 else 0.0
        if p == Y[0, i]:
            success_counter += 1

    return success_counter / m


data = read_csv('diabetes.csv')
# data = read_csv('logistic-regression.csv')

reg_lambda_value = 1e-2

# plot_learning_curves(data['X_train'], data['Y_train'], data['X_test'], data['Y_test'], reg_lambda_value)

params = minimize_cost(data['X_train'], data['Y_train'], reg_lambda_value)

print('============')

print('train precision = {}'.format(calculate_precision(data['X_train'], data['Y_train'], params['W'], params['b'])))
print('test precision = {}'.format(calculate_precision(data['X_test'], data['Y_test'], params['W'], params['b'])))
print('train cost = {}'.format(cost(data['X_train'], data['Y_train'], params['W'], params['b'])))
print('test cost = {}'.format(cost(data['X_test'], data['Y_test'], params['W'], params['b'])))

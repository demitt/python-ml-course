import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, W, B):
    return sigmoid(np.dot(W, X) + B)


def cost(X, W, B, Y):
    hypothesis_tr = h(X, W, B).T
    m = X.shape[1]
    value = (- np.dot(Y, np.log(hypothesis_tr + 10E-10)) - np.dot((1 - Y), np.log(1 - hypothesis_tr + 10E-10))) / m
    return value[0, 0]


def cost_gradients(X, W, B, Y):
    delta = h(X, W, B) - Y
    m = X.shape[1]

    return dict(
        W=np.dot(delta, X.T) / m,
        B=np.sum(delta) / m
    )


def minimize_cost(X, Y, alpha, iterations_count):
    n = X.shape[0]

    W = np.zeros((1, n))
    B = np.zeros((1, 1))

    for iteration in range(iterations_count):
        gradients = cost_gradients(X, W, B, Y)
        W = W - alpha * gradients['W']
        B = B - alpha * gradients['B']
        # print(cost(X, W, B, Y))

    return dict(W=W, B=B)


# Return:
#       matrix m*(n + 1)
def read_data(file_name):
    return np.genfromtxt(file_name, delimiter=',')


# TODO description
# Args:
#       data - matrix m*(n + 1),
#       train_data_part - part of train data, value in range (0, 1).
# Return:
#       dict('train', 'test'), each of them - dict('X', 'Y'),
#           X is matrix  n*train_data_count,
#           Y is matrix  1*train_data_count,
def prepare_samples(data, train_data_part):
    np.random.shuffle(data)
    data = data.T

    X_values = data[0:-1, :]  # n*m (in our case n == 1)
    Y_values = data[-1:, :]  # 1*m

    m = data.shape[1]
    train_data_count = int(train_data_part * m)

    train_data = dict(
        X=X_values[:, 0:train_data_count],
        Y=Y_values[:, 0:train_data_count]
    )
    test_data = dict(
        X=X_values[:, train_data_count:],
        Y=Y_values[:, train_data_count:]
    )

    return dict(
        train=train_data,
        test=test_data
    )


def normalize_dataset(X):
    min_value = np.min(X, axis=1, keepdims=True)
    max_value = np.max(X, axis=1, keepdims=True)
    normalized = (X - min_value) / (max_value - min_value)
    return dict(
        X=normalized,
        min=min_value,
        max=max_value
    )


def denormalize_parameters(W, B, min_value, max_value):
    W_denormalized = W / (max_value - min_value).T
    B_denormalized = B - np.dot(W, min_value / (max_value - min_value))
    return W_denormalized, B_denormalized


def map_probability_to_class(probability):
    # We use ">=" to avoid points on the border line;
    # we just decided: "if point is on the line therefore it belongs to '1.0' class"
    return 1.0 if probability >= 0.5 else 0.0


#
#
data = read_data('logistic-regression.csv')
samples = prepare_samples(data, train_data_part=0.7)

train_data = samples['train']
test_data = samples['test']
X = train_data['X']
Y = train_data['Y']

normalized = normalize_dataset(X)
X_normalized = normalized['X']

alpha = 0.1
iterations_count = 5000
params = minimize_cost(X_normalized, Y, alpha, iterations_count)
W, B = (params['W'], params['B'])

W, B = denormalize_parameters(W, B, normalized['min'], normalized['max'])
print('Params: W = {}, B = {}, min cost = {}'.format(W, B, cost(X, W, B, Y)))

W_ = W / np.abs(B)
B_ = B / np.abs(B)
print('Params: W = {}, B = {}, min cost = {}'.format(W_, B_, cost(X, W_, B_, Y)))

print('X-border (in case of 1 feature) = {}'.format(round((-B / W)[0, 0], 2)))

X_test = test_data['X']
Y_test = test_data['Y']
Y_probability_predicted = h(X_test, W, B)
m_test = Y_test.shape[1]
success_counter = 0
print('Y | Y_pred. | X\n--|---------|-------')
for i in range(m_test):
    probability_predicted = Y_probability_predicted[0, i]
    class_predicted = map_probability_to_class(probability_predicted)
    is_success = int(class_predicted) == int(Y_test[0, i])
    print('{} | {}\t| {}   {}'.format(
        Y_test[0, i],
        class_predicted,
        X_test[0, i],
        ('' if is_success else 'NOT')
    ))
    if is_success:
        success_counter += 1

success_percentage = round(success_counter / m_test * 100, 2)
print('Success predictions: {}% ({}/{})'.format(success_percentage, success_counter, m_test))

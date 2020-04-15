import numpy as np

import dao
from regression_functions import h, cost, minimize_cost


# Args:
#       data - matrix m*(n + 1),
#       train_data_part - part of train data, value in range (0, 1).
# Return:
#       dict('train', 'test'), each of them - dict('X', 'Y'),
#           X - matrix (n * train_data_count),
#           Y - matrix (1 * train_data_count),
def prepare_samples(data, train_data_part):
    np.random.shuffle(data)
    data = data.T

    X_values = data[0:-1, :]  # (n * m) (in our case n == 1)
    Y_values = data[-1:, :]  # (1 * m)

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


def analyse_and_print_results(data, W, B):
    X = data['X']
    Y = data['Y']
    Y_probability_predicted = h(X, W, B)

    m = X.shape[1]
    success_counter = 0

    print('\nResult:')
    print('Y   |Y_pred.|X')
    print('----|-------|-------')

    for i in range(m):
        probability_predicted = Y_probability_predicted[0, i]
        class_predicted = map_probability_to_class(probability_predicted)
        is_success = int(class_predicted) == int(Y[0, i])
        print('{} | {} \t| {}   {}'.format(
            Y[0, i],
            class_predicted,
            X[0, i],
            ('' if is_success else 'NOT')
        ))
        if is_success:
            success_counter += 1

    success_percentage = round(success_counter / m * 100, 2)
    print('Success predictions: {}% ({}/{})'.format(success_percentage, success_counter, m))


def map_probability_to_class(probability):
    # We use ">=" to avoid points on the border line;
    # we just decided: "if point is on the line therefore it belongs to '1.0' class"
    return 1.0 if probability >= 0.5 else 0.0


def main(alpha, iterations_count, data_file_name, train_data_part):
    # Reading and splitting datasets:
    data = dao.read_data(data_file_name)
    samples = prepare_samples(data, train_data_part=train_data_part)
    X = samples['train']['X']
    Y = samples['train']['Y']

    # Dataset normalization:
    normalized = normalize_dataset(X)

    # Cost fn minimization:
    params = minimize_cost(normalized['X'], Y, alpha, iterations_count)
    W, B = (params['W'], params['B'])

    # Parameters denormalization:
    W, B = denormalize_parameters(W, B, normalized['min'], normalized['max'])
    print('Params: W = {}, B = {}'.format(W, B))

    # Just test print:
    W_ = W / np.abs(B)
    B_ = B / np.abs(B)
    print('Params: W = {}, B = {}, min cost = {}'.format(W_, B_, cost(X, W_, B_, Y)))
    print('X-border (in case of 1 feature) = {}'.format(round((-B / W)[0, 0], 2)))

    # Results:
    analyse_and_print_results(samples['test'], W, B)


main(alpha=0.1, iterations_count=5000, data_file_name='logistic-regression.csv', train_data_part=0.7)

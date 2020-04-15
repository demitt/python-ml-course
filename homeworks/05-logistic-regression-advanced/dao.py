import numpy as np


# Return:
#       matrix m*(n + 1)  , n - number of features
def read_data(file_name):
    return np.genfromtxt(file_name, delimiter=',')

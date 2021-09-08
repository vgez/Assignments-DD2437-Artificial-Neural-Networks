import numpy as np


def data_init():

    x = np.arange(0, (2*np.pi), 0.1)
    x_test = np.arange(0.05, (2*np.pi), 0.1)

    t_sin = np.array(np.sin(2*x))
    t_sin_test = np.array(np.sin(2*x_test))

    t_sq = np.where(t_sin >= 0, 1, -1)
    t_sq_test = np.where(t_sin_test >= 0, 1, -1)

    return x, x_test, t_sin, t_sin_test, t_sq, t_sq_test


def rbf_init(size, var):
    rbf_nodes = []
    weights = []

    for i in range(size):
        rbf_nodes.append(np.random.normal(
            loc=np.random.uniform(0, (2*np.pi)), scale=var))
        weights.append(np.random.normal(loc=0, scale=var))

    return np.array(rbf_nodes), np.array(weights)


def batch_training(rbf, w, x, y):
    pass


r, w = rbf_init(3, 0.1)
print(r)
print(w)

import numpy as np
from multivariatenormal import multivariatenormal
from activations import sigmoid


def deltasinglelayerseq(x, label, weights):
    # Bias is an extra row in x and weights
    # and is not added by this function.
    return weights


def deltasinglelayerbatch(data, labels, weights):
    # Bias is an extra row in x and weights
    # and is not added by this function.
    eta = 0.001
    activations = np.matmul(weights, data)
    sigmoidv = np.vectorize(sigmoid)
    res = sigmoidv(activations)
    deltaw = -eta*np.matmul(np.matmul(weights, data) - labels, np.transpose(data))
    print(deltaw)
    return weights + deltaw

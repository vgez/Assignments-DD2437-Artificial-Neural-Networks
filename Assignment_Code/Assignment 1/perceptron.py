import matplotlib.pyplot as plt
import numpy as np
from multivariatenormal import multivariatenormal


def stepfunction(x):
    if x > 0:
        return -1
    elif x < 0:
        return 1
    else:
        return 0


def perceptronsinglelayerbatch(data, labels, weights, outputs=1, epochs=20):
    # Bias is an extra row in x and weights
    # and is not added by this function.
    eta = 0.1
    activations = np.matmul(weights, data)
    res = np.where(activations > 0, 1, -1)
    deltaw = eta*data*(labels-res)
    return weights - np.mean(deltaw, axis=1)


def perceptronsinglelayerseq(x, label, weights):
    # Bias is an extra row in x and weights
    # and is not added by this function.

    learningrate = 0.001

    # Multiplies the weights with the input data.
    res = np.multiply(weights, x)
    deltares = res[0]+res[1]+res[2]
    if deltares > 0:
        y = -1
    elif deltares < 0:
        y = 1
    else:
        return weights

    # Getting error
    e = label - y

    out = weights+e*x*learningrate
    return out

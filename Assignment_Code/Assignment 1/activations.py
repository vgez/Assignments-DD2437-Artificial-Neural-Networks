import numpy as np
def sigmoid(x):
    return 2.0/(1.0+np.exp(-x))-1.0

def sigmoid_d(x):
    return ((1.0+sigmoid(x)*(1.0-sigmoid(x)))/2.0)

def step(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

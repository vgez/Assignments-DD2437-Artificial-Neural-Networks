import numpy as np
from pprint import pprint
import math


class Neural_net:

    # setup vector holds the network structure, nodes in each layer
    def __init__(self, setup, theta, std):

        # class attributes
        self.weights_layers = []
        self.theta = theta
        self.classified_correct = 0

        # initialize weight layers randomly
        for i in range(len(setup) - 1):
            self.weights_layers.append(
                np.random.normal(0, std, (setup[i + 1], setup[i] + 1))
            )

    def backprop(self, X, T):
        # updates sequentially
        sq_errors = []
        for i in range(np.size(X, 1)):
            # pprint(X[i])
            x = np.array([X[:, i]])
            t = T[i]
            node_layers = self.forward_pass(x)
            deltas, sq_err = self.backward_pass(node_layers, t)
            sq_errors.append(sq_err)
            self.weight_update(node_layers, deltas)
        return sq_errors

    def forward_pass(self, x):

        # add bias to x
        def add_bias(x):
            x = np.append(x, 1.0)
            return np.array([x]).transpose()

        # convert x to a float, -1 <= x <= 1
        def sigmoid(x):
            return (2 / (1 + math.exp(-x))) - 1

        node_layers = []
        sigmoid_v = np.vectorize(sigmoid)

        x = add_bias(x)
        node_layers.append(x)

        for wL in self.weights_layers:
            x = np.matmul(wL, x)
            x = sigmoid_v(x)
            x = add_bias(x.transpose())
            node_layers.append(x)

        # remove bias from output layer
        node_layers[-1] = np.delete(node_layers[-1], np.size(x, 0) - 1, axis=0)
        return node_layers

    def backward_pass(self, node_layers, t):

        # derivitive of sigmoid() in forward_pass()
        def sigmoid_der(x):
            return ((1 + x) * (1 - x)) / 2

        deltas = []
        sigmoid_der_v = np.vectorize(sigmoid_der)

        if np.all(np.sign(node_layers[-1]), t):
            self.classified_correct += 1

        sq_err = np.square(node_layers[-1] - t)

        delta = np.matmul((node_layers[-1] - t),
                          sigmoid_der_v(node_layers[-1]))
        deltas.insert(0, delta)

        for i, l in enumerate(reversed(self.weights_layers[1:])):

            current_nodes = node_layers[(-(i + 2))]
            delta = np.multiply(l.transpose() * delta,
                                sigmoid_der_v(current_nodes))
            delta = delta[:-1, :]
            deltas.insert(0, delta)

        return deltas, sq_err

    def weight_update(self, nodes, deltas):
        for i, d in enumerate(deltas):
            delta_layer = -np.matmul((self.theta * d), nodes[i].transpose())
            self.weights_layers[i] = np.add(
                self.weights_layers[i], delta_layer)

    def clear_classified(self):
        self.classified_correct = 0

    def get_classified(self):
        return self.classified_correct

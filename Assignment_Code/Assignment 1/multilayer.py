import numpy as np
import matplotlib.pyplot as plt
import random


def initialize_neural_network(inputs, neuron_structure):
    # num_of_inputs = neuron_structure[0]
    # num_of_outputs = neuron_structure[-1]
    all_weights = []
    all_layers = []
    all_layers.append(inputs)
    for i in range(len(neuron_structure) - 1):
        all_weights.append(np.random.rand(
            neuron_structure[i+1], neuron_structure[i] + 1))

    for i in range(len(neuron_structure) - 2):
        hidden_layer = np.array(
            [0 for neuron in range(neuron_structure[i+1])] + [1])
        all_layers.append(hidden_layer)

    output_layer = np.array([0 for neuron in range(neuron_structure[-1])])
    all_layers.append(output_layer)

    return all_layers, all_weights


def activation_function():
    pass


def forward_phase():
    pass


def backward_phase():
    pass


def backpropagation(layers, weights):

    print(layers)
    print(weights)


def main():

    # generation of non-linearly separable data
    n = 2
    mean_A = [-1.5, -1.5]
    mean_B = [1.5, 1.5]
    cov = [[0.5, 0], [0, 0.5]]

    x1, y1 = np.random.multivariate_normal(mean_A, cov, n).T

    x2, y2 = np.random.multivariate_normal(mean_B, cov, n).T

    shuffled_x = np.concatenate((x1, x2)), np.concatenate((y1, y2))

    shuffled_x = np.array(shuffled_x)

    random.Random(4).shuffle(shuffled_x[0])
    random.Random(4).shuffle(shuffled_x[1])
    targets = [1 for i in range(n)] + [-1 for i in range(n)]

    random.Random(4).shuffle(targets)

    shuffled_x = np.append(shuffled_x, [[1]*2*n], axis=0)

    all_layers, all_weights = initialize_neural_network(
        shuffled_x, [2, 4, 1])

    backpropagation(all_layers, all_weights)


if __name__ == "__main__":
    main()

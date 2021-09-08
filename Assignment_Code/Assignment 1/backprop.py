import numpy as np
import random
from matplotlib import pyplot as plt
import single_layer_datagen as datagen
import plots as plot


class ANN():

    def __init__(self, inputs, neuron_structure, targets):
        self.inputs = inputs
        self.neuron_structure = neuron_structure
        self.targets = targets
        self.classifications = 0
        self.err_values = []
        self.outputs = []
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        for i in range(len(self.neuron_structure) - 1):
            weights.append(np.random.normal(loc=0, scale=0.5, size=(
                self.neuron_structure[i+1], self.neuron_structure[i])))

        return weights

    def activation(self, z):
        return (2.0/(1.0+np.exp(-z)))-1.0

    def activation_prime(self, z):
        return (((1.0+z)*(1.0-z))/2.0)

    def loss_function(self, a, t):
        c = 0.5*(sum((a-t)**2))
        return c

    def add_bias(self, a):
        a = np.append(a, 1.0)
        return np.array([a]).transpose()

    def forward_phase(self):
        for i in range(len(self.inputs[0])):
            network_layers = []
            vectorized_activation = np.vectorize(self.activation)

            current_input = self.inputs[:, i]
            current_input = self.add_bias(current_input)
            network_layers.append(current_input)

            for weight_layer in self.weights:
                z = np.matmul(weight_layer, current_input)
                a = vectorized_activation(z)
                a = self.add_bias(a.transpose())
                network_layers.append(a)
                current_input = a

            network_layers[-1] = np.delete(network_layers[-1],
                                           np.size(current_input, 0) - 1, axis=0)

            # start backward phase
            self.backward_phase(network_layers, i)

    def backward_phase(self, network_layers, i):
        deltas = []
        vectorized_activation_prime = np.vectorize(self.activation_prime)
        self.outputs.append(network_layers[-1])
        if(len(network_layers[-1]) > 1):
            for j, node in enumerate(network_layers[-1]):
                if(np.sign(node) == np.sign(self.targets[j])):
                    self.classifications += 1

            current_delta = np.matmul(
                (network_layers[-1] - self.targets), vectorized_activation_prime(network_layers[-1]))
        else:
            if(np.sign(network_layers[-1]) == np.sign(self.targets[i])):
                self.classifications += 1

            self.err_values.append(self.loss_function(
                network_layers[-1], self.targets[i]))

            current_delta = np.matmul(
                (network_layers[-1] - self.targets[i]), vectorized_activation_prime(network_layers[-1]))
        deltas.insert(0, current_delta)

        for network_layer, weight_layer in enumerate(reversed(self.weights[1:])):
            current_network_layer = network_layers[-(network_layer + 2)]
            current_delta = np.multiply(np.matmul(weight_layer.transpose(
            ), current_delta), vectorized_activation_prime(current_network_layer))
            current_delta = current_delta[:-1, :]
            deltas.insert(0, current_delta)

        self.update(network_layers, deltas)

    def update(self, network_layers, deltas):
        negative_learning_rate = -0.005
        for i, delta in enumerate(deltas):
            current_deltas = np.matmul(
                (negative_learning_rate * delta), network_layers[i].transpose())
            self.weights[i] = np.add(self.weights[i], current_deltas)


def main():

    length, shuffled_x, targets, w = datagen.data_non_sep_subsampled('25each')
    #shuffled_x, targets = datagen.create_data_enc()
    neural_network = ANN(shuffled_x, [2, 4, 1], np.array(targets))
    mean_squared_errors = []
    correct_classifications = []

    for epoch in range(100):
        neural_network.classifications = 0
        neural_network.outputs = []
        neural_network.forward_phase()
        print("Epoch " + str(epoch) + ": " +
              str(np.mean(neural_network.err_values)) + ", correct classifications: " + str(neural_network.classifications))
        mean_squared_errors.append(np.mean(neural_network.err_values))
        correct_classifications.append(neural_network.classifications)
    """ plot.get_3D_plot(x, y, z)
    #plot.get_3D_plot(x, y, np.reshape(neural_network.outputs, n*n))

    plot.get_3D_plot(x, y, np.reshape(neural_network.outputs, (n, n)))
    plt.show() """


if __name__ == "__main__":
    main()

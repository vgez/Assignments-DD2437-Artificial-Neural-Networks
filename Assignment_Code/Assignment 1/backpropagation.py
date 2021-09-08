import numpy as np
import random
from matplotlib import pyplot as plt


class ANN():

    def __init__(self, inputs, num_of_outputs, num_of_hidden_layers, neuron_structure, targets):
        self.inputs = inputs
        self.num_of_outputs = num_of_outputs
        self.num_of_hidden_layers = num_of_hidden_layers
        self.neuron_structure = neuron_structure
        self.targets = targets
        self.a_values = []
        self.z_values = []
        self.c_values = []
        self.err_values = []
        self.derivatives = []
        self.classifications = 0
        self.all_layers, self.all_weights = self.initialize_neural_network()

    def initialize_neural_network(self):
        # num_of_inputs = neuron_structure[0]
        # num_of_outputs = neuron_structure[-1]
        all_weights = []
        all_layers = []
        all_layers.append(self.inputs)
        for i in range(len(self.neuron_structure) - 1):
            all_weights.append(np.random.normal(scale=0.5, size=(
                self.neuron_structure[i+1], self.neuron_structure[i] + 1)))

        for i in range(len(self.neuron_structure) - 2):
            hidden_layer = np.array(
                [0 for neuron in range(self.neuron_structure[i+1])] + [1])
            all_layers.append(hidden_layer)

        output_layer = np.array(
            [0 for neuron in range(self.neuron_structure[-1])])
        all_layers.append(output_layer)

        return all_layers, all_weights

    def get_outputs(self):
        output_values = []
        for i in range(len(self.a_values)):
            output_values.append(
                self.a_values[i][self.num_of_hidden_layers + 1][0].tolist())
        return [j for sub in output_values for j in sub]

    def sigmoid(self, z):
        return (2.0/(1.0+np.exp(-z)))-1.0

    def sigmoid_prime(self, z):
        return (((1.0+z)*(1.0-z))/2.0)

    def loss_function(self, a, i):
        c = 0.5*(sum((a-self.targets[i])**2))
        return c

    def loss_function_prime(self, a, i):
        c_prime = sum((a-self.targets[i]))
        return c_prime

    def add_bias(self, a):
        li = []
        for i in a:
            li.append(i)
        bias = np.array([1.0])
        li.append(bias)
        a = np.array(li)
        return a

    def forward_phase(self):
        for i in range(len(self.inputs[0])):
            z_current = []
            a_current = []
            current_input = np.array([[self.inputs[0][i]], [
                                     self.inputs[1][i]], [self.inputs[2][i]]])
            a_current.append(current_input)
            # print(current_input)
            for j in range(len(self.neuron_structure) - 1):
                z = np.matmul(self.all_weights[j], current_input)
                z_current.append(z)
                a = self.sigmoid(z)
                if(j < len(self.neuron_structure) - 2):
                    a = self.add_bias(a)
                a_current.append(a)
                current_input = a

            self.a_values.append(a_current)
            self.z_values.append(z_current)
            # calculate loss
            c = self.loss_function(a, j)
            self.c_values.append(c)

            # start backward phase
            self.backward_phase(a_current, i)

    def backward_phase(self, a, i):
        output_errors = (a[-1] - self.targets[i])
        self.err_values.append(self.loss_function(a[-1], i))
        if(np.sign(a[-1]) == np.sign(self.targets[i])):
            self.classifications += 1

            # all_derivatives = []
            # for i in range(len(self.targets)):
        input_derivatives = []
        for j in reversed(range(len(self.neuron_structure) - 1)):

            if(j == len(self.neuron_structure) - 2):
                # print(self.all_weights[j])
                layer_derivates = []
                for k in range(len(self.all_weights[j][0])):
                    derivative = output_errors * \
                        self.sigmoid_prime(self.a_values[i][j][0])
                    layer_derivates.append(derivative)
                input_derivatives.insert(0, layer_derivates)
            else:
                layer_derivatives = []
                derivative_sum = 0
                for k in range(len(layer_derivates[0])):
                    derivative_sum += layer_derivates[0][k] * \
                        self.all_weights[j+1][0][k]
                for k in range(len(self.all_weights[j])):
                    layer_in_layer_derivatives = []
                    for l in range(len(self.all_weights[j][k])):
                        derivative = derivative_sum * \
                            self.sigmoid_prime(self.a_values[i][j][l])
                        layer_in_layer_derivatives.append(derivative)
                    layer_derivatives.append(layer_in_layer_derivatives)
                input_derivatives.insert(0, layer_derivatives)

        self.update(input_derivatives)

    def update(self, input_derivatives):
        learning_rate = 0.001
        all_deltas = []
        for i in range(len(input_derivatives)):
            layer_deltas = []
            for j in range(len(input_derivatives[i])):
                if(i == len(input_derivatives) - 1):
                    delta_weights = -learning_rate * \
                        np.multiply(
                            input_derivatives[i], self.a_values[0][i][0])
                    layer_deltas.append(delta_weights)
                    break
                else:
                    delta_weights = -learning_rate * \
                        np.multiply(
                            input_derivatives[i][j], self.a_values[0][i])
                    layer_deltas.append(delta_weights)
            all_deltas.append(layer_deltas)

        for i in range(len(all_deltas)):
            for j in range(len(all_deltas[i])):
                self.all_weights[i][j] = self.all_weights[i][j] + \
                    all_deltas[i][j][0]


def create_data_sep():
    # generation of non-linearly separable data
    n = 50
    mean_A = [2.0, 2.0]
    mean_B = [-1.0, -1.0]
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
    return shuffled_x, targets


def create_data_xor(mean1=[10, 10], mean2=[0, 0], mean3=[0, 10], mean4=[10, 0], cov=[[0.10, 0], [0, 0.5]], n=5):
    norm1 = np.random.multivariate_normal(mean1, cov, n).T
    norm2 = np.random.multivariate_normal(mean2, cov, n).T
    norm3 = np.random.multivariate_normal(mean3, cov, n).T
    norm4 = np.random.multivariate_normal(mean4, cov, n).T

    shuffledx = np.concatenate((norm1[0], norm2[0], norm3[0], norm4[0])), np.concatenate(
        (norm1[1], norm2[1], norm3[1], norm4[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    T = makelabels(n, -1) + makelabels(n, -1) + \
        makelabels(n, 1) + makelabels(n, 1)
    random.Random(4).shuffle(T)
    shuffledx = np.append(shuffledx, [[1]*4*n], axis=0)
    return shuffledx, T


def makelabels(n, label):
    labels = [label for i in range(n)]
    return labels


def main():

    shuffled_x, targets = create_data_sep()

    neural_network = ANN(shuffled_x, 1, 1, [
                         2, 10, 1], np.array(targets))
    # print(neural_network.all_weights)
    mean_squared_errors = []
    correct_classifications = []
    for epoch in range(1000):
        neural_network.classifications = 0
        neural_network.forward_phase()
        print("Epoch " + str(epoch) + ": " +
              str(np.mean(neural_network.err_values)) + ", correct classifications: " + str(neural_network.classifications))
        mean_squared_errors.append(np.mean(neural_network.err_values))
        correct_classifications.append(neural_network.classifications)
    # print(shuffled_x)
    plt.plot(mean_squared_errors)
    plt.plot(correct_classifications)
    plt.show()


if __name__ == "__main__":
    main()

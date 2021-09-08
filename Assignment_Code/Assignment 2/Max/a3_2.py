import numpy as np
import os
import matplotlib.pyplot as plt
from rbfnetwork import RadialBasisFunctionNetwork


def sin(x):
    y = np.sin(2 * x)
    return y


def square(x):
    y = np.where(sin(x) >= 0, 1, -1)
    return y


def main():
    input_nodes_n = 1
    rbf_nodes_n = 20
    output_nodes_n = 1
    x_max = 2 * np.pi
    step_size = 0.1

    FIGURE_DIR = "figures/a3_2/{}nodes/".format(rbf_nodes_n)
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)

    print("!!! a3_2 starting! !!!")
    data = np.arange(0, x_max, step_size)
    noise = np.random.normal(0, step_size, data.size)
    # targets_square_noise = square(data) + noise
    # targets_sin_noise = sin(data) + noise
    targets_sin = sin(data)
    targets_square = square(data)

    test_data = np.arange(0.05, x_max, 0.1)
    test_targets_square = square(test_data)
    test_targets_sin = sin(test_data)
    print("Data size:", data.shape)

    model_square = RadialBasisFunctionNetwork(input_nodes_n, rbf_nodes_n, output_nodes_n)
    model_square.fit_sequential(data, targets_square)
    predictions_square = model_square.predict_sequential(test_data)

    model_sin = RadialBasisFunctionNetwork(input_nodes_n, rbf_nodes_n, output_nodes_n)
    model_sin.fit_sequential(data, targets_sin)
    predictions_sin = model_sin.predict_sequential(test_data)

    print("incorrect classifications (square):", np.count_nonzero(np.sign(predictions_square) - test_targets_square))
    print("total error (square):", np.sum(np.square(predictions_square - test_targets_square)))
    print("total error (sin):", np.sum(np.square(predictions_sin - test_targets_sin)))


    plt.figure(3)
    plt.plot(test_data, test_targets_square)
    rbf_placements_square = model_square.get_rbf_node_placement()
    plt.plot(rbf_placements_square, np.zeros(len(rbf_placements_square)), "r*")
    plt.plot(test_data, predictions_square, "g")
    plt.grid()
    plt.savefig(FIGURE_DIR+"test_square_wave.png")

    plt.figure(4)
    plt.plot(test_data, test_targets_sin)
    rbf_placements_sinus = model_sin.get_rbf_node_placement()
    plt.plot(rbf_placements_sinus, np.zeros(len(rbf_placements_sinus)), "r*")
    plt.plot(test_data, predictions_sin, "g")
    plt.grid()
    plt.savefig(FIGURE_DIR+"test_sinus_wave.png")

    plt.figure(5)
    plt.plot(model_sin.total_error_list, label="Sinus wave training error")
    plt.plot(model_square.total_error_list, label="Square wave training error")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
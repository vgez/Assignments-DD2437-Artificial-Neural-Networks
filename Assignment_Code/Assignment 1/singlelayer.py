import numpy as np
import matplotlib.pyplot as plt
import random
import single_layer_datagen as datagen


def check_misclassifications(targets, y):
    misclass = 0
    for i in range(len(targets)):
        if (np.sign(targets[i]) != np.sign(y[i])):
            misclass += 1
    return misclass


def check_sens_spec(targets, y, method, length):
    #a is positives, b is negatives
    true_positives = 0
    true_negatives = 0

    def evaluate_total(method, length):
        return {
            '25each': [75, 75],
            '50a': [50, 100],
            '50b': [100, 50],
            '2080': [length - 100, 100],
            'none': [100, 100]
        }[method]

    total_samples_list = evaluate_total(method, length)
    for i in range(len(targets)):
        if (np.sign(targets[i]) == np.sign(y[i])):
            if(targets[i] == 1):
                true_positives += 1
            else:
                true_negatives += 1
    return true_positives/total_samples_list[0], true_negatives/total_samples_list[1]


def train_perceptron_single_layer_batch(inputs, weights, targets, num_of_epochs, learning_rate, stop_early, method, length):
    sens = []
    spec = []
    mean_err = []
    for epoch in range(num_of_epochs):
        y_prime = np.matmul(np.transpose(weights), inputs)
        y = np.where(y_prime > 0, 1, -1)
        sq_err = (y_prime - targets)**2
        mean_err.append(np.mean(sq_err))
        true_positives, true_negatives = check_sens_spec(
            targets, y, method, length)
        sens.append(true_positives)
        spec.append(true_negatives)
        delta_weights = -learning_rate*inputs*(y-targets)
        weights = weights + np.mean(delta_weights, axis=1)
        if(epoch % (num_of_epochs/10) == 0):
            plot_decision_boundary(
                inputs, weights, method, epoch/num_of_epochs)
    plt.show()
    return sens, spec


def train_delta_single_layer_seq(inputs, weights, targets, num_of_epochs, learning_rate, stop_early):
    mean_err = []
    misclassifications = []
    for epoch in range(num_of_epochs):
        epoch_err = 0
        for i in range(len(inputs[0])):
            err = np.dot(weights, inputs[:, i]) - targets[i]
            sq_err = err**2
            delta_weights = -learning_rate*err*inputs[:, i]
            weights = weights + delta_weights
            epoch_err += sq_err
        y_prime = np.matmul(np.transpose(weights), inputs)
        y = np.where(y_prime > 0, 1, -1)
        misclass = check_misclassifications(targets, y)
        misclassifications.append(misclass)
        mean_err.append(epoch_err / len(inputs[0]))
        if(epoch % (num_of_epochs/10) == 0):
            plot_decision_boundary(
                inputs, weights, '2080', epoch/num_of_epochs)
    plt.show()
    return mean_err, misclassifications


def train_delta_single_layer_batch(inputs, weights, targets, num_of_epochs, learning_rate, stop_early):
    misclassifications = []
    mean_err = []
    for epoch in range(num_of_epochs):
        y_prime = np.matmul(weights, inputs)
        # y = np.where(y_prime > 0, 1, -1)
        # check_misclassifications(targets, y)
        err = y_prime - targets
        sq_err = err**2
        y = np.where(y_prime > 0, 1, -1)
        misclass = check_misclassifications(targets, y)
        misclassifications.append(misclass)
        delta_weights = -learning_rate*np.multiply(err, inputs)
        weights = weights + np.mean(delta_weights, axis=1)
        if(epoch % (num_of_epochs/10) == 0):
            plot_decision_boundary(
                inputs, weights, '2080', epoch/num_of_epochs)
        mean_err.append(np.mean(sq_err))
    plt.show()
    return mean_err, misclassifications


def plot_decision_boundary(data_matrix, weights, method, opacity):

    def pick_method(method):
        return {
            'none': [['b+' for i in range(100)], ['ro' for i in range(100)]],
            '25each': [['b+' for i in range(75)], ['ro' for i in range(75)]],
            '50a': [['b+' for i in range(50)], ['ro' for i in range(100)]],
            '50b': [['b+' for i in range(100)], ['ro' for i in range(50)]],
            '2080': [['b+' for i in range(len(data_matrix[0]) - 100)], ['ro' for i in range(100)]]
        }[method]

    line_length = 10
    decision_boundary = [[] for i in range(2)]
    x = np.linspace(-5, 5, 100)
    if(len(weights) == 3):
        y = -((weights[0]*x)/weights[1]+weights[2]/weights[1])
    else:
        y = -((weights[0]*x)/weights[1])
    decision_boundary[0][:] = x
    decision_boundary[1][:] = y

    colors = pick_method(method)

    colors = np.concatenate((colors[0], colors[1]))
    random.Random(4).shuffle(colors)

    for i in range(len(data_matrix[0])):
        plt.plot(data_matrix[0][i], data_matrix[1][i], colors[i])

    plt.plot(decision_boundary[0],
             decision_boundary[1], 'black', alpha=opacity)
    plt.xlim(-(line_length/2), line_length/2)
    plt.ylim(-(line_length/2), line_length/2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.0001)


def plot_error(err, err_title):
    plt.plot(err[0])
    plt.plot(err[1])
    plt.title(err_title)
    plt.show()


def plot_err_val(x, err, res, err_val=None, res_val=None):
    fig, (ax1, ax2) = plt.subplots(2)
    for val in err:
        ax1.plot(x, val)
    for val in res:
        ax2.plot(x, val)
    if not err_val == None:
        ax1.plot(x, err_val)
    if not res_val == None:
        ax2.plot(x, res_val)
    #ax1.set_title("Mean square error")
    #ax2.set_title("Unclassified data/All data")
    ax1.legend(('random 25% from each class', 'random 50% from class A',
                'random 50% from class B', '20% / 80% subsets from class A', 'No subsampling'))
    ax2.legend(('random 25%% from each class', 'random 50% from class A',
                'random 50% from class B', '20% / 80% subsets from class A', 'No subsampling'))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Number of Misclassifications")
    plt.show()


def plot_sens_spec(x, sens, spec, err_val=None, res_val=None):
    fig, (ax1, ax2) = plt.subplots(2)
    for val in sens:
        ax1.plot(x, val)
    for val in spec:
        ax2.plot(x, val)
    if not err_val == None:
        ax1.plot(x, err_val)
    if not res_val == None:
        ax2.plot(x, res_val)
    #ax1.set_title("Mean square error")
    #ax2.set_title("Unclassified data/All data")
    ax1.legend(('random 25% from each class', 'random 50% from class A',
                'random 50% from class B', '20% / 80% subsets from class A', 'No subsampling'))
    ax2.legend(('random 25%% from each class', 'random 50% from class A',
                'random 50% from class B', '20% / 80% subsets from class A', 'No subsampling'))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("True Positives")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("True Negatives")
    plt.show()


def plot_err_one(x, err, res, err_val=None, res_val=None):
    fig, (ax1, ax2) = plt.subplots(2)
    for i in range(len(res)):
        ax1.plot(x, err[0])
        ax2.plot(x, res[0])
    if not err_val == None:
        ax1.plot(x, err_val)
    if not res_val == None:
        ax2.plot(x, res_val)
    #ax1.set_title("Mean square error")
    #ax2.set_title("Unclassified data/All data")
    ax1.legend(['Delta Rule'])
    ax2.legend(['Delta Rule'])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Number of Misclassifications")
    plt.show()


def main():
    l1, inputs1, targets1, weights1 = datagen.data_non_sep_subsampled('25each')
    l2, inputs2, targets2, weights2 = datagen.data_non_sep_subsampled('50a')
    l3, inputs3, targets3, weights3 = datagen.data_non_sep_subsampled('50b')
    l4, inputs4, targets4, weights4 = datagen.data_non_sep_subsampled('2080')
    inputs5, targets5, weights5 = datagen.data_non_sep()
    epochs = 2000

    #mean_err_p, misclass_p = train_perceptron_single_layer_batch(inputs, weights, targets, epochs, 0.001, True)
    sens_d1, spec_d1 = train_perceptron_single_layer_batch(
        inputs1, weights1, targets1, epochs, 0.001, True, '25each', 150)
    sens_d2, spec_d2 = train_perceptron_single_layer_batch(
        inputs2, weights2, targets2, epochs, 0.001, True, '50a', 150)
    sens_d3, spec_d3 = train_perceptron_single_layer_batch(
        inputs3, weights3, targets3, epochs, 0.001, True, '50b', 150)
    sens_d4, spec_d4 = train_perceptron_single_layer_batch(
        inputs4, weights4, targets4, epochs, 0.001, True, '2080', l4)
    sens_d5, spec_d5 = train_perceptron_single_layer_batch(
        inputs5, weights5, targets5, epochs, 0.001, True, 'none', 200)

    #mean_err_seq, misclass_seq = train_delta_single_layer_seq(inputs, weights, targets, epochs, 0.0005, True)

    """ plot_err_val([i for i in range(epochs)], [mean_err_d1, mean_err_d2, mean_err_d3, mean_err_d4,
                                              mean_err_d5], [misclass_d1, misclass_d2, misclass_d3, misclass_d4, misclass_d5]) """

    plot_sens_spec([i for i in range(epochs)], [sens_d1, sens_d2, sens_d3,
                                                sens_d4, sens_d5], [spec_d1, spec_d2, spec_d3, spec_d4, spec_d5])


if __name__ == "__main__":
    main()

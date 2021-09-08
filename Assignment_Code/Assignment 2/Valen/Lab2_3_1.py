import numpy as np
import matplotlib.pyplot as plt
import random
from rbf_net import RBF
import inspect


AVG_RNDS = 5


def plot_graphs(range, inlist, legends, ylabel=None, xlabel=None):
    plt.figure()
    for l in inlist:
        plt.plot(range, l)
    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_prediction(error_list, model_list, legends, xlabel, range=np.arange(0, (2*np.pi), 0.1)):
    """
    makes predictions for network with lowest error and plots predictions and ground truth
    """
    lowest_score_index = error_list.index(min(error_list))
    best_model = model_list[lowest_score_index]
    gt, pred = best_model.validate()

    plt.figure()
    for l in [gt, pred]:
        plt.plot(range, l)
    plt.legend(legends)
    plt.xlabel(xlabel + str(lowest_score_index + 1))


def err_to_num_rbfs(inlist, batch, noise, eta=None, sin_data=True, var=0.4):
    """
    creates network for node config in inlist, returns list of errors and 
    list of networks
    """
    error_list = []
    rbf_nets = []
    for num_nodes in inlist:
        errors_tmp = []
        nets_tmp = []
        for avg_rnd in range(AVG_RNDS):
            RBF_net = RBF(1, num_nodes, 1, 0, (2*np.pi), var)
            err_test = RBF_net.train(batch=batch, eta=eta, noise=noise, sin_data=sin_data)
            errors_tmp.append(err_test)
            nets_tmp.append(RBF_net)
        best_net = nets_tmp[errors_tmp.index(min(errors_tmp))]
        error_list.append(sum(errors_tmp) / AVG_RNDS)
        rbf_nets.append(best_net)
    return error_list, rbf_nets


def err_to_width_eta(data, inlist, batch, noise):
    """
    creates network for var/eta values in inlist, returns list of errors and 
    list of networks
    """
    error_list = []
    rbf_nets = []
    for item in inlist:
        var = item if batch else 0.1
        eta = item if not batch else None
        RBF_net = RBF(1, 22, 1, 0, (2*np.pi), var)
        err_test = RBF_net.train(batch=batch, eta=eta, noise=noise)
        error_list.append(err_test)
        rbf_nets.append(RBF_net)
    return error_list, rbf_nets


def three_one(num_nodes_list):
    """
    BATCH LEARNING num of rbf units
    """
    # calculate error for number of nodes in hidden layer, batch, no noise
    err_num_nodes_batch_sin, nets_num_nodes_batch_sin = err_to_num_rbfs(num_nodes_list, batch=True, noise=False)
    err_num_nodes_batch_sq, nets_num_nodes_batch_sq = err_to_num_rbfs(
        num_nodes_list, batch=True, noise=False, sin_data=False)

    # calculate error for number of nodes in hidden layer, batch, noise
    err_num_nodes_batch_sin_n, nets_num_nodes_batch_sin_n = err_to_num_rbfs(num_nodes_list, batch=True, noise=True)
    err_num_nodes_batch_sq_n, nets_num_nodes_batch_sq_n = err_to_num_rbfs(
        num_nodes_list, batch=True, noise=True, sin_data=False)

    # check units for error below 0.1, 0.01 and 0.001 respectively
    def unit_check(error_list, name):
        one = False
        zero_one = False
        zero_zero_on = False
        print('error list:', name)
        for i, error in enumerate(error_list):
            if error < 0.1 and not one:
                print('RBF-units used for an error below 0.1:', i+1)
                one = True
            elif error < 0.01 and not zero_one:
                print('RBF-units used for an error below 0.01:', i+1)
                zero_one = True
            elif error < 0.001 and not zero_zero_on:
                print('RBF-units used for an error below 0.001:', i+1)
                zero_zero_on = True

    unit_check(err_num_nodes_batch_sin, 'sin(2x)')
    unit_check(err_num_nodes_batch_sq, 'square(2x)')

    # plot and predict sin, no noise
    plot_prediction(err_num_nodes_batch_sin, nets_num_nodes_batch_sin, [
                    'ground truth', 'prediction'], 'Predicting sin(2x), batch learning, RBF units: ')

    # plot and predict square, no noise
    plot_prediction(err_num_nodes_batch_sq, nets_num_nodes_batch_sq, [
                    'ground truth', 'prediction'], 'Predicting square(2x), batch learning, RBF units: ')

    # plot and predict sin, noise
    plot_prediction(err_num_nodes_batch_sin_n, nets_num_nodes_batch_sin_n, ['ground truth', 'prediction'],
                    'Predicting sin(2x) with noise, batch learning, RBF units: ')

    # plot and predict square, noise
    plot_prediction(err_num_nodes_batch_sq_n, nets_num_nodes_batch_sq_n, ['ground truth', 'prediction'],
                    'Predicting square(2x) with noise, batch learning, RBF units: ')

    # plot residual error depending on num of nodes in hidden layer, batch
    plot_graphs(num_nodes_list, [err_num_nodes_batch_sin, err_num_nodes_batch_sq,
                                 err_num_nodes_batch_sin_n, err_num_nodes_batch_sq_n],
                ['sin2x', 'square2x', 'sin2x noise', 'square2x noise'],
                'Residual error', 'Number of RBF units, batch learning')


def check_seq(num_nodes_list):
    """
    SEQUENTIAL LEARNING num of rbf units
    """
    eta = 0.3
    var = 0.01
    # calculate error for number of nodes in hidden layer, sequential, no noise
    err_num_nodes_seq_sin, nets_num_nodes_seq_sin = err_to_num_rbfs(
        num_nodes_list, batch=False, noise=False, eta=eta, var=var)
    err_num_nodes_seq_sq, nets_num_nodes_seq_sq = err_to_num_rbfs(
        num_nodes_list, batch=False, noise=False, eta=eta, sin_data=False, var=var)

    # calculate error for number of nodes in hidden layer, sequential, noise
    err_num_nodes_seq_sin_n, nets_num_nodes_seq_sin_n = err_to_num_rbfs(
        num_nodes_list, batch=False, noise=True, eta=eta, var=var)
    err_num_nodes_seq_sq_n, nets_num_nodes_seq_sq_n = err_to_num_rbfs(
        num_nodes_list, batch=False, noise=True, eta=eta, sin_data=False, var=var)

    # plot and predict sin, no noise
    plot_prediction(err_num_nodes_seq_sin, nets_num_nodes_seq_sin, [
                    'ground truth', 'prediction'], 'Predicting sin(2x), sequential learning, RBF units: ')

    # plot and predict square, no noise
    plot_prediction(err_num_nodes_seq_sq, nets_num_nodes_seq_sq, [
                    'ground truth', 'prediction'], 'Predicting square(2x), sequential learning, RBF units: ')

    # plot and predict sin, noise
    plot_prediction(err_num_nodes_seq_sin_n, nets_num_nodes_seq_sin_n, ['ground truth', 'prediction'],
                    'Predicting sin(2x) with noise, sequential learning, RBF units: ')

    # plot and predict square, noise
    plot_prediction(err_num_nodes_seq_sq_n, nets_num_nodes_seq_sq_n, ['ground truth', 'prediction'],
                    'Predicting square(2x) with noise, sequential learning, RBF units: ')

    # plot residual error depending on num of nodes in hidden layer, sequential
    plot_graphs(num_nodes_list, [err_num_nodes_seq_sin, err_num_nodes_seq_sq, err_num_nodes_seq_sin_n, err_num_nodes_seq_sq_n], ['sin2x', 'square2x',
                                                                                                                                 'sin2x noise', 'square2x noise'], 'Residual error', 'Number of RBF units, sequential learning')


if __name__ == '__main__':
    # num_nodes
    num_nodes_list = np.arange(1, 25)
    three_one(num_nodes_list)
    # check_seq(num_nodes_list)
    plt.show()

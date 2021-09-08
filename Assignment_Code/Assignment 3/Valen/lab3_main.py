from matplotlib import pyplot as plt
import numpy as np
from hopfield_net import Hopfield_network


def data_loader():
    return np.loadtxt('annda_lab3/pict.dat', dtype=int, delimiter=',').reshape(-1, 1024)


def task_3_1():
    # input patterns
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1]).reshape(1, -1)

    x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1]).reshape(1, -1)

    hop_net = Hopfield_network(x1.shape[1], [x1, x2, x3], asynch=False)


def task_3_2():
    # data
    data = data_loader()
    N = data.shape[1]
    p1 = np.array(data[0]).reshape(1, -1)
    p2 = np.array(data[1]).reshape(1, -1)
    p3 = np.array(data[2]).reshape(1, -1)
    p10 = np.array(data[9]).reshape(1, -1)
    p11 = np.array(data[10]).reshape(1, -1)
    # training
    training_samples = data[:3]
    hopfield = Hopfield_network(N, training_samples, asynch=True)

    # recall stored patterns
    p1_recall, _, _ = hopfield.update_nodes(p1)
    p2_recall, _, _ = hopfield.update_nodes(p2)
    p3_recall, _, _ = hopfield.update_nodes(p3)
    print('p1 recalled correctly:', np.count_nonzero(p1_recall - p1) == 0)
    print('p2 recalled correctly:', np.count_nonzero(p2_recall - p2) == 0)
    print('p3 recalled correctly:', np.count_nonzero(p3_recall - p3) == 0)

    # recall distorted patterns
    p10_recall, _, _ = hopfield.update_nodes(p10)
    p10_recall_rand, _, _ = hopfield.update_nodes(p10, random=True, store_progress=True)
    p11_recall, _, _ = hopfield.update_nodes(p11)
    p11_recall_rand, _, _ = hopfield.update_nodes(p11, random=True, store_progress=True)
    print('p1 corrupted recalled correctly (ordered update):', np.count_nonzero(p10_recall - p1) == 0)
    print('p1 corrupted recalled correctly (random update):', np.count_nonzero(p10_recall_rand - p1) == 0)
    print('p2 corrupted recalled correctly (ordered update):', np.count_nonzero(p11_recall - p2) == 0)
    print('p2 corrupted recalled correctly (random update):', np.count_nonzero(p11_recall_rand - p2) == 0)
    print(np.count_nonzero(p10_recall - p1))
    print(np.count_nonzero(p10_recall_rand - p1))

    p11_progress = hopfield.get_progress()

    plt.imshow(p11_progress[0].reshape((32, 32)), cmap="gray")
    plt.show()


if __name__ == "__main__":
    task_3_2()

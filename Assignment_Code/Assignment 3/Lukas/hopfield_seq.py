import numpy as np
import random as rnd
from matplotlib import pyplot as plt


class HopfieldNetwork():

    def __init__(self, N, INpatterns, asynch=True):
        self.asynch = asynch
        self.N = N
        self.init_patterns = INpatterns
        self.W = np.zeros((self.N, self.N), dtype=int)
        self.calc_weights()

    def calc_weights(self):
        identity = np.identity(self.N, dtype=int)
        for pattern in self.init_patterns:
            self.W += (np.matmul(pattern.T, pattern) - identity)

    def update_nodes(self, INpattern, iterations=10, random=False):
        print(INpattern.shape)
        current = INpattern
        stable_count = 0

        if self.asynch:
            order = np.arange(self.N)
            while iterations > 0:
                rnd.shuffle(order) if random else None
                for i in order:
                    current_node = self.W[:, [i]]
                    result = np.sign(np.dot(current_node.T, current.T))
                    print(result)
                    if current[:, i] != result:
                        current[:, i] = result
                        stable_count = 0
                    stable_count += 1
                iterations -= 1
                if stable_count >= 2*self.N:
                    recall = current
                    break
        else:
            while iterations > 0:
                recall = np.sign(np.matmul(current, self.W))
                if (recall == current).all() and stable_count >= 5:
                    break
                stable_count += 1 if (recall == current).all() else 0
                current = recall
                iterations -= 1
        return recall, stable_count, iterations

    def get_weights(self):
        return self.W


def main():
    # input patterns
    data = np.loadtxt('pict.dat', dtype=int,
                      delimiter=',').reshape(-1, 1024)
    patterns = [data[i] for i in range(data.shape[0])]

    hopfield = HopfieldNetwork(1024, patterns[0:3])
    plt.imshow(patterns[0].reshape((32, 32)), cmap="gray")

    recall, _, _ = hopfield.update_nodes(patterns[0].reshape(1, 1024))
    plt.imshow(patterns[0].reshape((32, 32)), cmap="gray")

    plt.imshow(recall.reshape((32, 32)), cmap="gray")
    plt.show()

    """ img_pattern = patterns[0].reshape((32, 32))
    plt.imshow(img_pattern, cmap="gray")
    plt.show()
    print(data.shape)
    print(patterns) """


"""     hop_net = Hopfield_network(x_1.shape[1], [x_1, x_2, x_3], asynch=True)

    x_1d = np.array([1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x_2d = np.array([1, 1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x_3d = np.array([1, 1, 1, -1, 1, 1, -1, 1]).reshape(1, -1)
    print(x_3d)
    tot_iter = 10
    recall, stable_count, iterations = hop_net.update_nodes(x_3d, iterations=tot_iter)
    print(x_3d)
    print('pattern:', x_3d, 'recall:', recall, 'stable_count:', stable_count, 'iterations:', tot_iter - iterations) """


if __name__ == "__main__":
    main()

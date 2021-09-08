import numpy as np
import random as rnd


class Hopfield_network():

    def __init__(self, N, INpatterns, asynch=True):
        self.asynch = asynch
        self.N = N
        self.init_patterns = INpatterns
        self.W = np.zeros((self.N, self.N), dtype=int)
        self._calc_weights()
        self.latest_progress = None

    def _calc_weights(self):
        identity = np.identity(self.N, dtype=int)
        for pattern in self.init_patterns:
            pattern = pattern.reshape(1, self.N)
            self.W += (np.matmul(pattern.T, pattern) - identity)

    def update_nodes(self, INpattern, iterations=30, random=False, store_progress=False, store_inter=20):
        current = INpattern
        stable_count = 0

        if self.asynch:
            self.latest_progress = []
            self.latest_progress.append(current)
            order = np.arange(self.N)
            rnd.shuffle(order) if random else None
            while iterations > 0:
                for i in order:
                    current_node = self.W[:, [i]]
                    result = np.sign(np.dot(current_node.T, current.T))
                    if current[:, i] != result:
                        current[:, i] = result
                        stable_count = 0
                    stable_count += 1
                    self.latest_progress.append(current) if (i % store_inter == 0) and store_progress else None
                iterations -= 1
                recall = current
                if stable_count >= 2*self.N:
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

    def calc_energy(self, pattern):
        energy_sum = 0
        n = self.weights.shape[0]
        for i in range(n):
            for j in range(n):
                energy_sum += self.weights[i, j]*pattern[:, i]*pattern[:, j]
        return -energy_sum

    def calc_energy_fast(self, pattern):
        outer = np.outer(pattern, np.transpose(pattern))
        energy_sum = np.sum(np.multiply(self.weights, outer))
        return np.multiply(-1, energy_sum)

    def calc_energy_3(self, pattern):
        return -np.dot(pattern, self.weights).dot(pattern.T)

    def get_weights(self):
        return self.W

    def get_progress(self):
        return self.latest_progress

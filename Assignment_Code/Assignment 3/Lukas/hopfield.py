import numpy as np
from matplotlib import pyplot as plt
import itertools
import random


class HopfieldNet():

    def __init__(self, patterns, symmetric, asynch=True):
        self.patterns = patterns
        self.symmetric = symmetric
        self.weights = self.initialize_weights()
        self.asynch = asynch
        self.energy = []

    def initialize_weights(self, random_gauss=False, symmetric=True):
        if random_gauss:
            if random_gauss:
                dimension = self.patterns[0].shape[1]
                w = np.random.normal(0, 1, (dimension, dimension))
                np.fill_diagonal(w, 0)
                if self.symmetric:
                    w = np.multiply(0.5, np.add(w, w.T))
                return w
        else:
            w_tot = []
            for p in range(self.patterns.shape[0]):
                w_curr = np.matmul(self.patterns[p].T, self.patterns[p])
                """ w_curr = np.subtract(w_curr, np.identity(
                    w_curr.shape[0], dtype=int)) """
                w_tot.append(w_curr)

            w_sum = np.empty(shape=w_curr.shape)
            for i in range(len(w_tot)):
                if i == 0:
                    w_sum = w_tot[i]
                else:
                    w_sum = np.add(w_tot[i], w_sum)
            return w_sum

    def recall(self, pattern, rand=True, calc_energy=False, calc_stable=True):
        if self.asynch:
            pre_pattern = pattern
            stable_count = 0
            iterations = 5
            is_stable = False
            order = np.arange(pattern.shape[1])
            if rand:
                random.shuffle(order)
            while (iterations > 0):
                for i in order:
                    curr = self.weights[:, [i]]
                    result = np.sign(np.dot(curr.T, pre_pattern.T))
                    if pre_pattern[:, i] != result:
                        pre_pattern[:, i] = result
                        stable_count = 0
                    stable_count += 1
                    if calc_energy:
                        self.energy.append(self.calc_energy_fast(pre_pattern))
                if stable_count >= 100:
                    if calc_energy:
                        return pre_pattern, self.energy
                    else:
                        is_stable = True
                        return pre_pattern, is_stable
                iterations -= 1

        else:
            pre_pattern = pattern
            self.energy.append(self.calc_energy_fast(pre_pattern))
            for i in range(100):
                after_pattern = np.sign(np.matmul(pre_pattern, self.weights))
                if ((pre_pattern == after_pattern).all()):
                    return after_pattern, self.energy
                else:
                    self.energy.append(self.calc_energy_fast(after_pattern))
                    pre_pattern = after_pattern
            return pre_pattern, self.energy
        if calc_energy:
            return pre_pattern, self.energy
        else:
            return pre_pattern, is_stable

    def attractors(self, all_patterns):
        attractors = []
        for pattern in all_patterns:
            pre_pattern = pattern
            for i in range(30):
                after_pattern = np.sign(np.matmul(pre_pattern, self.weights))
                if ((pre_pattern == after_pattern).all()):
                    attractors.append(after_pattern)
                else:
                    pre_pattern = after_pattern
        return attractors

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


def task3_1():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1]).reshape(1, -1)

    x = np.array([x1, x2, x3])

    hopfield = HopfieldNet(x)

    # Assert that the net has stored the three patterns
    """ np.testing.assert_array_equal(x1, hopfield.batch_test(x1))
    np.testing.assert_array_equal(x2, hopfield.batch_test(x2))
    np.testing.assert_array_equal(x3, hopfield.batch_test(x3)) """

    x1_d = np.array([1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
    x2_d = np.array([1, 1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
    x3_d = np.array([1, 1, 1, -1, 1, 1, -1, 1]).reshape(1, -1)

    """ np.testing.assert_array_equal(x1, hopfield.batch_test(x1_d))
    np.testing.assert_array_equal(x2, hopfield.batch_test(x2_d))
    np.testing.assert_array_equal(x3, hopfield.batch_test(x3_d)) """

    """ print(hopfield.batch_test(x1_d))
    print(hopfield.batch_test(x2_d)) """
    # print(hopfield.recall(x3_d))

    """ all_patterns = []
    lst = list(itertools.product([-1, 1], repeat=8))

    for i in range(len(lst)):
        all_patterns.append(list(lst[i]))

    # all_patterns = all_patterns.astype(int)

    # print(len(all_patterns))

    attract = hopfield.attractors(all_patterns)
    unique = np.unique(attract, axis=0)
    print(unique)
    print(unique.shape[0]) """
    x1_daf = np.array([-1, 1, -1, -1, 1, 1, -1, 1]).reshape(1, -1)
    x2_daf = np.array([1, -1, 1, -1, 1, 1, 1, -1]).reshape(1, -1)
    x3_daf = np.array([-1, 1, -1, 1, 1, -1, 1, 1]).reshape(1, -1)

    print(hopfield.recall(x3_daf))


def task3_2():
    data = np.loadtxt('pict.dat', dtype=int,
                      delimiter=',').reshape(-1, 1024)

    p1 = np.array(data[0]).reshape(1, -1)
    p2 = np.array(data[1]).reshape(1, -1)
    p3 = np.array(data[2]).reshape(1, -1)
    p10 = np.array(data[9]).reshape(1, -1)
    p11 = np.array(data[10]).reshape(1, -1)

    plt.imshow(p11.reshape((32, 32)), cmap="gray")
    plt.show()

    x = np.array([p1, p2, p3])
    hopfield = HopfieldNet(x)

    plt.imshow(p11.reshape((32, 32)), cmap="gray")
    plt.show()

    recall_1 = hopfield.recall(p1)
    recall_2 = hopfield.recall(p2)
    recall_3 = hopfield.recall(p3)
    # recall_10 = hopfield.recall(p10)
    # recall_11 = hopfield.recall(p11)

    plt.imshow(p11.reshape((32, 32)), cmap="gray")
    plt.show()

    # plt.imshow(p1.reshape((32, 32)), cmap="gray")
    # plt.show()

    """ plt.imshow(recall_1.reshape((32, 32)), cmap="gray")
    plt.show()
    plt.imshow(recall_2.reshape((32, 32)), cmap="gray")
    plt.show()
    plt.imshow(recall_3.reshape((32, 32)), cmap="gray")
    plt.show() """

    """ plt.imshow(recall_10.reshape((32, 32)), cmap="gray")
    plt.show()
    plt.imshow(recall_11.reshape((32, 32)), cmap="gray")
    plt.show() """


def task3_3():
    data = np.loadtxt('pict.dat', dtype=int,
                      delimiter=',').reshape(-1, 1024)

    p1 = np.array(data[0]).reshape(1, -1)
    p2 = np.array(data[1]).reshape(1, -1)
    p3 = np.array(data[2]).reshape(1, -1)
    p10 = np.array(data[9]).reshape(1, -1)
    p11 = np.array(data[10]).reshape(1, -1)

    x = np.array([p1, p2, p3])

    hopfield_1 = HopfieldNet(x, False)
    hopfield_2 = HopfieldNet(x, True)

    # Calculating energies
    """ p1_e = hopfield.calc_energy_fast(p1)
    p2_e = hopfield.calc_energy_fast(p2)
    p3_e = hopfield.calc_energy_fast(p3)
    p10_e = hopfield.calc_energy_fast(p10)
    p11_e = hopfield.calc_energy_fast(p11)

    print(p1_e)
    print(p2_e)
    print(p3_e)
    print(p10_e)
    print(p11_e) """

    # recall_1, energy_list1 = hopfield.recall(p1)
    # recall_2, energy_list2 = hopfield.recall(p2)
    # recall_3, energy_list3 = hopfield.recall(p3)
    # recall_10, energy_list10 = hopfield.recall(p10)
    # recall_11, energy_list11 = hopfield.recall(p11)

    # Plotting energy after every iteration
    """ plt.imshow(recall_1.reshape((32, 32)), cmap="gray")
    plt.show()
    plt.plot(energy_list1, color="green", label="Pattern 1")
    plt.ylabel("Energy")
    plt.xlabel("Iterations")
    plt.show() """

    # Experiments with random weight matrix, does not converge
    arbitrary_start = np.random.choice([1, -1], (1, 1024))
    recall_non, energy_list_non = hopfield_1.recall(arbitrary_start)
    recall_sym, energy_list_sym = hopfield_2.recall(arbitrary_start)
    plt.plot(np.arange(1024*5), energy_list_non, label="Non-symmetric matrix")
    plt.plot(np.arange(1024*5), energy_list_sym, label="Symmetric matrix")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.legend(loc="best")
    plt.show()


def task3_4():

    def add_noise(pattern, percent):
        amount_of_noise = percent * pattern.shape[1]
        noise_pattern = np.copy(pattern)
        indices = random.sample(
            list(np.arange(pattern.shape[1])), int(amount_of_noise))
        for i in indices:
            noise_pattern[:, i] = -1 * pattern[:, i]
        return noise_pattern

    data = np.loadtxt('pict.dat', dtype=int,
                      delimiter=',').reshape(-1, 1024)

    p1 = np.array(data[0]).reshape(1, -1)
    p2 = np.array(data[1]).reshape(1, -1)
    p3 = np.array(data[2]).reshape(1, -1)
    p10 = np.array(data[9]).reshape(1, -1)
    p11 = np.array(data[10]).reshape(1, -1)

    unique, counts = np.unique(p3, return_counts=True)
    print(dict(zip(unique, counts)))
    x = np.array([p1, p2, p3])

    hopfield = HopfieldNet(x)

    p1_count = np.zeros(11)
    p2_count = np.zeros(11)
    p3_count = np.zeros(11)

    for i in range(11):
        for j in range(50):
            noise = i / 10
            p1_n = add_noise(p1, noise)
            p2_n = add_noise(p2, noise)
            p3_n = add_noise(p3, noise)

            recall_p1, _ = hopfield.recall(p1_n)
            recall_p2, _ = hopfield.recall(p2_n)
            recall_p3, _ = hopfield.recall(p3_n)

            if ((recall_p1 == p1).all()):
                p1_count[i] += 1
            if ((recall_p2 == p2).all()):
                p2_count[i] += 1
            if ((recall_p3 == p3).all()):
                p3_count[i] += 1

        p1_count[i] = p1_count[i] / 50
        p2_count[i] = p2_count[i] / 50
        p3_count[i] = p3_count[i] / 50

    print(p1_count)
    print(p2_count)
    print(p3_count)
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.plot(noise_levels, p1_count, label="Pattern 1")
    plt.plot(noise_levels, p2_count, label="Pattern 2")
    plt.plot(noise_levels, p3_count, label="Pattern 3")
    plt.legend(loc='best')
    plt.xlabel("Amount of noise")
    plt.ylabel("Ratio of correct convergence (50 iterations)")
    plt.show()

    """ p1_noisy_10 = add_noise(p1, 0.1)
    p1_noisy_20 = add_noise(p1, 0.2)
    p1_noisy_30 = add_noise(p1, 0.3)
    p1_noisy_40 = add_noise(p1, 0.4)
    p1_noisy_50 = add_noise(p1, 0.5)

    recall_p1_10, _ = hopfield.recall(p1_noisy_10)
    recall_p1_20, _ = hopfield.recall(p1_noisy_20)
    recall_p1_30, _ = hopfield.recall(p1_noisy_30)
    recall_p1_40, _ = hopfield.recall(p1_noisy_40)
    recall_p1_50, _ = hopfield.recall(p1_noisy_50)

    print((recall_p1_10 == p1).all())

    plt.imshow(recall_p1_10.reshape((32, 32)), cmap="gray")
    plt.title("10% noise")
    plt.show()

    plt.imshow(recall_p1_20.reshape((32, 32)), cmap="gray")
    plt.title("20% noise")
    plt.show()

    plt.imshow(recall_p1_30.reshape((32, 32)), cmap="gray")
    plt.title("30% noise")
    plt.show()

    plt.imshow(recall_p1_40.reshape((32, 32)), cmap="gray")
    plt.title("40% noise")
    plt.show()

    plt.imshow(recall_p1_50.reshape((32, 32)), cmap="gray")
    plt.title("50% noise")
    plt.show()

    # p2
    p2_noisy_10 = add_noise(p2, 0.1)
    p2_noisy_20 = add_noise(p2, 0.2)
    p2_noisy_30 = add_noise(p2, 0.3)
    p2_noisy_40 = add_noise(p2, 0.4)
    p2_noisy_50 = add_noise(p2, 0.5)

    recall_p2_10, _ = hopfield.recall(p2_noisy_10)
    recall_p2_20, _ = hopfield.recall(p2_noisy_20)
    recall_p2_30, _ = hopfield.recall(p2_noisy_30)
    recall_p2_40, _ = hopfield.recall(p2_noisy_40)
    recall_p2_50, _ = hopfield.recall(p2_noisy_50)

    plt.imshow(recall_p2_10.reshape((32, 32)), cmap="gray")
    plt.title("10% noise")
    plt.show()

    plt.imshow(recall_p2_20.reshape((32, 32)), cmap="gray")
    plt.title("20% noise")
    plt.show()

    plt.imshow(recall_p2_30.reshape((32, 32)), cmap="gray")
    plt.title("30% noise")
    plt.show()

    plt.imshow(recall_p2_40.reshape((32, 32)), cmap="gray")
    plt.title("40% noise")
    plt.show()

    plt.imshow(recall_p2_50.reshape((32, 32)), cmap="gray")
    plt.title("50% noise")
    plt.show()

    # p3
    p3_noisy_10 = add_noise(p3, 0.1)
    p3_noisy_20 = add_noise(p3, 0.2)
    p3_noisy_30 = add_noise(p3, 0.3)
    p3_noisy_40 = add_noise(p3, 0.4)
    p3_noisy_50 = add_noise(p3, 0.5)

    recall_p3_10, _ = hopfield.recall(p3_noisy_10)
    recall_p3_20, _ = hopfield.recall(p3_noisy_20)
    recall_p3_30, _ = hopfield.recall(p3_noisy_30)
    recall_p3_40, _ = hopfield.recall(p3_noisy_40)
    recall_p3_50, _ = hopfield.recall(p3_noisy_50)

    plt.imshow(recall_p3_10.reshape((32, 32)), cmap="gray")
    plt.title("10% noise")
    plt.show()

    plt.imshow(recall_p3_20.reshape((32, 32)), cmap="gray")
    plt.title("20% noise")
    plt.show()

    plt.imshow(recall_p3_30.reshape((32, 32)), cmap="gray")
    plt.title("30% noise")
    plt.show()

    plt.imshow(recall_p3_40.reshape((32, 32)), cmap="gray")
    plt.title("40% noise")
    plt.show()

    plt.imshow(recall_p3_50.reshape((32, 32)), cmap="gray")
    plt.title("50% noise")
    plt.show() """


def task3_5():

    def random_patterns(num_of_patterns, dims):
        patterns = np.random.choice([1, -1], (num_of_patterns, dims))
        return patterns

    number_of_patterns = 200

    patterns = random_patterns(number_of_patterns, 100)
    ratio_of_stable_patterns = np.empty(number_of_patterns)
    ratio_of_correct_patterns = np.empty(number_of_patterns)

    for i in range(number_of_patterns):
        print(i)
        curr_patterns = patterns[:i+1]
        pattern_list = []
        for p in curr_patterns:
            pattern_list.append(np.array(p).reshape(1, -1))

        x = np.array(pattern_list)
        hopfield = HopfieldNet(x, True)
        stable_patterns = 0
        correct_patterns = 0
        for pattern in pattern_list:
            recall, stable = hopfield.recall(pattern)
            if stable:
                stable_patterns += 1
                if ((recall == pattern).all()):
                    correct_patterns += 1

        ratio = stable_patterns / (i+1)
        ratio_corr = correct_patterns / (i+1)
        ratio_of_stable_patterns[i] = ratio
        ratio_of_correct_patterns[i] = ratio_corr

    print(ratio_of_correct_patterns)
    plt.plot(ratio_of_correct_patterns)
    plt.show()

    print(ratio_of_stable_patterns)
    plt.plot(ratio_of_stable_patterns)
    plt.show()


if __name__ == "__main__":
    task3_5()

import numpy as np
import random
import math


def data_sep(mean1=[1.5, 1.0], mean2=[-1.5, -0.5], cov=[[0.5, 0], [0, 0.5]], n=100):
    norm1 = np.random.multivariate_normal(mean1, cov, n).T
    norm2 = np.random.multivariate_normal(mean2, cov, n).T
    shuffledx = np.concatenate(
        (norm1[0], norm2[0])), np.concatenate((norm1[1], norm2[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    targets = makelabels(n, 1) + makelabels(n, -1)
    random.Random(4).shuffle(targets)
    shuffledx = np.append(shuffledx, [[1]*2*n], axis=0)

    weights = [np.random.normal(scale=0.3), np.random.normal(
        scale=0.3), np.random.normal(scale=0.3)]

    weights_2 = [-0.5, 0.7, 0.2]

    return shuffledx, targets, weights


def data_sep_no_bias(mean1=[3.0, 3.0], mean2=[0.0, 0.0], cov=[[0.5, 0], [0, 0.5]], n=100):
    norm1 = np.random.multivariate_normal(mean1, cov, n).T
    norm2 = np.random.multivariate_normal(mean2, cov, n).T
    shuffledx = np.concatenate(
        (norm1[0], norm2[0])), np.concatenate((norm1[1], norm2[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    targets = makelabels(n, 1) + makelabels(n, -1)
    random.Random(4).shuffle(targets)

    weights = [np.random.normal(scale=0.3), np.random.normal(
        scale=0.3)]

    return shuffledx, targets, weights


def data_non_sep(mean1=[1.0, 0.3], mean2=[0.0, -0.1], cov=[[0.2, 0], [0, 0.3]], n=100):
    norm1 = np.random.multivariate_normal(mean1, cov, n).T
    norm2 = np.random.multivariate_normal(mean2, cov, n).T
    shuffledx = np.concatenate(
        (norm1[0], norm2[0])), np.concatenate((norm1[1], norm2[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    targets = makelabels(n, 1) + makelabels(n, -1)
    random.Random(4).shuffle(targets)
    shuffledx = np.append(shuffledx, [[1]*2*n], axis=0)

    weights_2 = [np.random.normal(scale=0.3), np.random.normal(
        scale=0.3), np.random.normal(scale=0.3)]

    weights = [0.1, -0.2, 0.1]

    return shuffledx, targets, weights


def data_non_sep_subsampled(method, mean1=[1.0, 0.3], mean2=[0.0, -0.1], cov=[[0.2, 0], [0, 0.3]], n=100):

    def adv_subsample(a, b):
        zipped_negative = []
        zipped_positive = []
        zipped_a = list(zip(a[0], a[1]))
        for val in zipped_a:
            if val[1] < 0:
                zipped_negative.append(val)
            else:
                zipped_positive.append(val)
        zipped_a = random.sample(zipped_negative, math.floor(len(zipped_negative) / 5)) + \
            random.sample(zipped_positive, math.floor(
                len(zipped_positive) * 0.8))
        return zipped_a + list(zip(b[0], b[1]))

    def pick_subsample(method, a, b):
        return {
            '25each': random.sample(list(zip(a[0], a[1])), 75) + random.sample(list(zip(b[0], b[1])), 75),
            '50a': random.sample(list(zip(a[0], a[1])), 50) + list(zip(b[0], b[1])),
            '50b': list(zip(a[0], a[1])) + random.sample(list(zip(b[0], b[1])), 50),
            '2080': adv_subsample(a, b)
        }[method]

    def create_targets(method, a, b):
        return {
            '25each': makelabels(75, 1) + makelabels(75, -1),
            '50a': makelabels(50, 1) + makelabels(100, -1),
            '50b': makelabels(100, 1) + makelabels(50, -1),
            '2080': makelabels((len(adv_subsample(a, b)) - 100), 1) + makelabels(100, -1)
        }[method]

    norm1 = np.random.multivariate_normal(mean1, cov, n).T
    norm2 = np.random.multivariate_normal(mean2, cov, n).T
    class_a = [norm1[0], norm1[1]]
    class_b = [norm2[0], norm2[1]]
    subsampling = pick_subsample(method, class_a, class_b)
    length = len(subsampling)
    input_x = [a[0] for a in subsampling]
    input_y = [a[1] for a in subsampling]
    inputs = [input_x, input_y]
    # random.Random(4).shuffle(inputs[0])
    # random.Random(4).shuffle(inputs[1])
    targets = create_targets(method, class_a, class_b)
    # random.Random(4).shuffle(targets)
    subsampling = np.append(inputs, [[1]*len(subsampling)], axis=0)
    weights = [np.random.normal(scale=0.3), np.random.normal(
        scale=0.3), np.random.normal(scale=0.3)]

    weights = [-0.5, 0.2, 0.3]

    return length, subsampling, targets, weights


def makelabels(n, label):
    labels = [label for i in range(n)]
    return labels

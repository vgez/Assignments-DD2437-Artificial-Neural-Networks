import numpy as np
import random


def create_data_sep():
    # generation of non-linearly separable data
    n = 100
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
    return shuffled_x, targets


def create_data_xor(mean1=[10, 10], mean2=[0, 0], mean3=[0, 10], mean4=[10, 0], cov=[[0.10, 0], [0, 0.5]], n=20):
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
    return shuffledx, T


def create_data_enc():
    x = [[[1.0], [-1.0], [-1.0], [-1.0],
          [-1.0], [-1.0], [-1.0], [-1.0]],
         [[-1.0], [1.0], [-1.0], [-1.0],
          [-1.0], [-1.0], [-1.0], [-1.0]],
         [[-1.0], [-1.0], [1.0], [-1.0],
          [-1.0], [-1.0], [-1.0], [-1.0]],
         [[-1.0], [-1.0], [-1.0], [1.0],
          [-1.0], [-1.0], [-1.0], [-1.0]],
         [[-1.0], [-1.0], [-1.0], [-1.0],
          [1.0], [-1.0], [-1.0], [-1.0]],
         [[-1.0], [-1.0], [-1.0], [-1.0],
          [-1.0], [1.0], [-1.0], [-1.0]],
         [[-1.0], [-1.0], [-1.0], [-1.0],
          [-1.0], [-1.0], [1.0], [-1.0]],
         [[-1.0], [-1.0], [-1.0], [-1.0],
          [-1.0], [-1.0], [-1.0], [1.0]]]
    x2 = [[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
          [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
          [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
          [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
          [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
          [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
          [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]]
    targets = [[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
               [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
               [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
               [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
               [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
               [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
               [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
               [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]]
    return x2, targets


def create_data_func_app():
    n = 50
    x_values = np.linspace(-5, 5, n)
    y_values = np.linspace(-5, 5, n)

    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z = np.exp(-((x_mesh**2+y_mesh**2)*0.1))-0.5

    data = np.array(
        [np.reshape(x_mesh, n*n), np.reshape(y_mesh, n*n)])

    targets = np.reshape(z, n*n)
    return data, targets, x_mesh, y_mesh, z, n


def create_data_chaotic():
    t = np.linspace(301, 1500, 1200)


def makelabels(n, label):
    labels = [label for i in range(n)]
    return labels


def main():
    create_data_chaotic()


if __name__ == "__main__":
    main()

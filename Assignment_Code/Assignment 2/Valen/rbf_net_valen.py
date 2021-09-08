import numpy as np
import random
from tqdm import tqdm

# for consitency
# np.random.seed(40)


class RBF():

    def __init__(self, input_size, rbf_size, output_size, lower_range, upper_range, var):

        self.input_size = input_size
        self.rbf_size = rbf_size
        self.output_size = output_size
        self.lower_range = lower_range
        self.upper_range = upper_range

        self.x_train = np.zeros((input_size, 1))
        self.x_test = np.zeros((input_size, 1))
        self.y_train = np.zeros((input_size, 1))
        self.y_test = np.zeros((input_size, 1))
        #self.weights = np.zeros((rbf_size, output_size))
        self.weights = np.random.normal(0, 0.1, size=(rbf_size, output_size))
        self.mu = None
        self.var = var
        self.eta = None
        self.batch = None

    def _calc_err_seq(self, f_hat, i):
        err = self.y_train[i] - f_hat
        return err

    def _calc_err(self, x_data, y_data):
        """
            Calculates phi matrix, multiplies by weights and returns average residual error.
            INPUTS: x_data (training samples), y_data (corresponding labels)
            OUTPUT: Average absolute error between predictions and labels 
        """
        phi = self._calc_phi(x_data, True)
        pred = np.dot(phi, self.weights)
        return np.mean(np.abs(y_data - pred)), pred

    def _update(self, err, phi):
        delta_w = self.eta * err * phi
        #self.weights = np.add(self.weights, delta_w)
        self.weights += delta_w

    def _gauss_rbf(self, mu, var, data):
        """
            Calculates and returns gaussian RBF kernel
            INPUTS: mu (position of RBF), var (variance of RBF)
            OUTPUT: Average absolute error between predictions and labels 
        """
        return np.exp((-(data-mu)**2)/(2*var))

    def set_data(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def _set_hyperparams(self, eta):
        self.eta = eta

    def _calc_phi(self, data, test=False):
        """
            Calls gauss_rbf to fill up phi matrix, which initiates as zeros of size (N, n)
            INPUTS: data (training samples)
            OUTPUT: phi (complete phi matrix)
        """
        if self.batch or test:
            phi = np.zeros((self.x_train.shape[0], self.rbf_size))
            for i in range(self.x_train.shape[0]):
                for j in range(self.rbf_size):
                    phi[i][j] = self._gauss_rbf(
                        self.mu[j], self.var, data[i])
        else:
            phi = np.zeros((self.rbf_size, 1))
            for i in range(self.rbf_size):
                phi[i] = self._gauss_rbf(self.mu[i], self.var, data)
        return phi

    def _forward_pass(self, i=None):
        """
            Forward pass for either ls-batch method or delta rule sequential approach.
            INPUTS: i (index in case of sequential learning)
            OUTPUT: -
        """

        if self.batch:
            phi = self._calc_phi(self.x_train)
            self.weights = np.dot(np.linalg.inv(
                np.dot(phi.T, phi)), np.dot(phi.T, self.y_train))
        else:
            phi = self._calc_phi(self.x_train[i])
            f_hat = np.dot(phi.T, self.weights)
            err = self._calc_err_seq(f_hat, i)
            self._update(err, phi)

    def validate(self):
        _, pred = self._calc_err(self.x_test, self.y_test)
        return self.y_test, pred

    def _set_mu(self, train_data, competitive):

        if competitive:
            iters = 100
            np.random.shuffle(train_data)

            # Initialize as random data sample to avoid dead unit problem
            positions = [train_data[np.random.randint(
                0, len(train_data) - 1)] for i in range(self.rbf_size)]

            for iter in range(iters):
                random_sample = train_data[np.random.randint(
                    0, len(train_data) - 1)]
                min_distance = np.finfo(np.float32).max
                min_index = 0
                for i, pos in enumerate(positions):
                    distance = np.linalg.norm(pos - random_sample)

                    if(distance < min_distance):
                        distance = min_distance
                        min_index = i

                delta_pos = self.eta * (random_sample-positions[min_index])
                positions[min_index] += delta_pos
        else:
            # even distribution of rbf nodes across input space
            positions = []
            for i in range(self.rbf_size):
                positions.append(((i+1) / (self.rbf_size+1)) * (2*np.pi))
        return positions

    def train(self, epochs=100, batch=True, eta=0.2, competitive=False, noise=False):
        """
            Main function for training process. 
            INPUTS: epochs (# of epochs if seq-learning), batch (bool if batch or not), eta (learning rate)
            OUTPUT: - 
        """
        self._set_hyperparams(eta)
        self.mu = self._set_mu(self.x_train, competitive)
        self.batch = batch

        if noise:
            # adding zero-mean additive gaussian noise (variance 0.1)
            gauss_noise = [random.gauss(0.0, 0.1)
                           for i in range(self.x_train.shape[0])]

            for i in range(self.x_train.shape[0]):
                self.x_train[i] += gauss_noise[i]
                self.x_test[i] += gauss_noise[i]

                # Add to targets as well?
                self.y_train[i] += gauss_noise[i]
                self.y_test[i] += gauss_noise[i]

        if not self.batch:
            for epoch in tqdm(range(epochs)):
                np.random.seed(epoch)
                np.random.shuffle(self.x_train)
                np.random.shuffle(self.y_train)
                for i in range(self.x_train.shape[0]):
                    self._forward_pass(i)
            err_train, _ = self._calc_err(self.x_train, self.y_train)
            err_test, _ = self._calc_err(self.x_test, self.y_test)
        else:
            self._forward_pass()
            err_train, _ = self._calc_err(self.x_train, self.y_train)
            err_test, _ = self._calc_err(self.x_test, self.y_test)
        return err_train, err_test

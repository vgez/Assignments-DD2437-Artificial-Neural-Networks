import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.spatial.distance import cityblock


class SOM():

    def __init__(self, data, weights, epochs, eta, hood_size):
        self.data = data
        self.weights = weights
        self.epochs = epochs
        self.eta = eta
        self.hood_size = hood_size

    def train(self, task=None):

        # determine change of hood size across epochs
        #hood_change = self.hood_size / self.epochs

        for epoch in range(self.epochs):
            self.hood_size = round(1*((self.epochs-epoch)/self.epochs))
            if task == 'cyclic':
                self.eta = self.eta * 0.99
            print("Epoch " + str(epoch+1) + " completed.")
            for i in range(self.data.shape[0]):
                winner = self.determine_winner(self.data[i])
                hood = self.determine_hood(winner, task, self.data[i])
                if task != 'votes':
                    self.update(hood, self.data[i])

        # extra loop through data to calculate winning node
        preds = self.predict()
        if task == "cyclicccc":
            # This will change
            tmp = np.vstack([self.weights[:, :], self.weights[0, :]])
            plt.figure()
            plt.plot(tmp.T[0], tmp.T[1], color='green', label='tour')
            plt.scatter(self.data.T[0], self.data.T[1],
                        color='red', label='data samples', alpha=0.8)
            plt.legend()
            plt.show()
        return preds, self.weights

    def determine_winner(self, sample):
        min_d = np.finfo(np.float32).max
        min_i = 0
        for i in range(self.weights.shape[0]):
            diff_vector = np.subtract(sample, self.weights[i])
            d = np.dot(diff_vector.T, diff_vector)
            if d < min_d:
                min_i = i
                min_d = d
        return min_i

    def determine_hood(self, index, task, curr_sample):
        if task == "cyclic":
            lower_bound = (index-self.hood_size) % 10
            upper_bound = (index + self.hood_size) % 10

            if lower_bound > upper_bound:
                overlap = True
                # switch values
                lower_bound, upper_bound = upper_bound, lower_bound
            else:
                overlap = False

            return [lower_bound, upper_bound, overlap]

        elif task == "votes":
            if len(str(index)) < 2:
                winner_row = 0
                winner_col = index
            else:
                winner_row = int(str(index)[0])
                winner_col = int(str(index)[1])

            for i in range(self.weights.shape[0]):
                if i < 10:
                    curr_row = 0
                    curr_col = i
                else:
                    curr_row = int(str(i)[0])
                    curr_col = int(str(i)[1])

                if cityblock([winner_row, winner_col], [curr_row, curr_col]) <= self.hood_size:
                    self.vote_update(curr_sample, i)
            return [None, None, None]
        else:
            lower_bound = max(0, (index-self.hood_size))
            upper_bound = min(self.weights.shape[0], (
                index+self.hood_size))
            return [lower_bound, upper_bound, False]

    def vote_update(self, sample, i):
        self.weights[i] = np.add(
            self.weights[i], (self.eta * np.subtract(sample, self.weights[i])))

    def update(self, hood, sample):
        start, end, overlap = hood[0], hood[1], hood[2]

        if overlap:
            # determine minus index of upper bound
            max_i = end - (self.weights.shape[0]+1)
            for i in range(start, max_i, -1):
                self.weights[i] = np.add(
                    self.weights[i], (self.eta * np.subtract(sample, self.weights[i])))
        else:
            for i in range(start, end):
                self.weights[i] = np.add(
                    self.weights[i], (self.eta * np.subtract(sample, self.weights[i])))

    def predict(self):
        preds = []

        for i in range(self.data.shape[0]):
            winner = self.determine_winner(self.data[i])
            preds.append([winner, i])

        return preds

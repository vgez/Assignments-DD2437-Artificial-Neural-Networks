import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import math
import random
import timeit
import time

learning_rate = 0.01
n = 10
beta = 0.2
tau = 25
gamma = 0.1
time_data = [1.5]
X = []
Y = []


def mackey_glass_time_series(t, t_25):
    return t + ((beta*t_25)/(1+t_25**n)) - gamma*t


def create_input_data(time, time_data):
    return np.array([time_data[time-20], time_data[time-15], time_data[time-10], time_data[time-5], time_data[time]])


for t in range(1525):
    time_data.append(mackey_glass_time_series(
        time_data[t], time_data[t-25] if t > 25 else 0))

for i in range(301, 1501):
    X.append(create_input_data(i, time_data))
    Y.append(time_data[i+5])

gauss_noise = [random.gauss(0.0, 0.09) for i in range(1200)]

for i in range(len(X)):
    X[i] += gauss_noise[i]
    Y[i] += gauss_noise[i]

X = np.array(X)
Y = np.array(Y)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1/6)
# print(len(train_x))
results_list = []
weight_list = []
print("hej")


def main():
    for i in range(10):
        mlp = MLPRegressor(hidden_layer_sizes=(
            6), learning_rate_init=learning_rate, max_iter=1000, solver='sgd', early_stopping=True, momentum=0.9, alpha=0.0001, verbose=False, learning_rate='constant')

        mlp.fit(train_x, train_y)
        y_pred = mlp.predict(test_x)
        loss = mean_squared_error(test_y, y_pred)
        results_list.append(loss)
        print(i+1)


start = time.time()
main()
print("%s seconds" % (time.time() - start))
print(str(max(results_list)) + ", " + str(min(results_list)) +
      ", " + str(sum(results_list) / 10))


# plt.plot(test_y)
# plt.plot(y_pred)
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.legend(["True Samples", "Predictions"])
# print(results_list)
# print(mlp.loss_)
# pd.DataFrame(time_data).plot()
# pd.DataFrame(mlp.loss_curve_).plot()
# plt.plot(mlp.loss_curve_)
# plt.ylabel("Error")
# plt.xlabel("Epoch")
# plt.legend(["Average Loss"])
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.legend("E")
# pd.DataFrame(gauss_noise).plot()
# plt.show()

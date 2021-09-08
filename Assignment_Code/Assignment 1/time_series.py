import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import math
from operator import itemgetter

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


for time in range(1525):
    time_data.append(mackey_glass_time_series(
        time_data[time], time_data[time-25] if time > 25 else 0))

for i in range(301, 1501):
    X.append(create_input_data(i, time_data))
    Y.append(time_data[i+5])

X = np.array(X)
Y = np.array(Y)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1/6)
# print(len(train_x))
results_list = []
weight_list = []
alpha_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.1, 0.01, 0.001,
                0.0001, 0.00001, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.1, 0.01, 0.001, 0.0001, 0.00001]
number_of_nodes = [6]

mlp = MLPRegressor(hidden_layer_sizes=(
    6), learning_rate_init=learning_rate, max_iter=1000, solver='sgd', early_stopping=True, momentum=0.9, alpha=0.0001, verbose=False, learning_rate='constant')


#sorted_results = sorted(results_list, key=itemgetter(2))

mlp.fit(train_x, train_y)
y_pred = mlp.predict(test_x)
loss = mean_squared_error(test_y, y_pred)


""" plt.plot(test_y)
plt.plot(y_pred)
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend(["True Samples", "Predictions"]) """

# pd.DataFrame(time_data).plot()
# pd.DataFrame(y_pred).plot()
# pd.DataFrame(mlp.loss_curve_).plot()
# plt.plot(mlp.loss_curve_)
plt.plot(test_y)
plt.plot(y_pred)
plt.ylabel("Error")
plt.xlabel("Epoch")
plt.legend(["Test labels", "Prediction on test set"])
# print(loss)
print(mlp.loss_)
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.legend("E")
plt.show()

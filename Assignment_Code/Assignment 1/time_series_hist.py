import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import math

learning_rate = 0.001
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

mlp = MLPRegressor(hidden_layer_sizes=(
    50), learning_rate_init=learning_rate, max_iter=1000, solver='adam', early_stopping=False, alpha=0.1, verbose=False, learning_rate='constant')

mlp.fit(train_x, train_y)
y_pred = mlp.predict(test_x)
loss = mean_squared_error(test_y, y_pred)


results_list.append([mlp.n_iter_, loss, mlp.loss_, mlp.coefs_])

num_bins = 8
plt.hist(np.array(mlp.coefs_).flatten(), bins=np.arange(-2.0, 2.2, 0.2) if i == 0 else 'auto', density=False,
         facecolor='blue', edgecolor='black', stacked=True)

plt.grid(axis='y', alpha=0.75)
plt.xlim(-2, 2)
plt.xlabel('Value', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('Histogram', fontsize=15)
plt.subplots_adjust(left=0.15)
plt.show()

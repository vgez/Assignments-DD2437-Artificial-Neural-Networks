import numpy as np
from matplotlib import pyplot as plt
from Lukas.rbf_net_lukas import RBF

ballist_train = np.loadtxt('Data/ballist.dat')
ballist_test = np.loadtxt('Data/balltest.dat')

x_ballist_train = ballist_train[:, 0:2]
y_ballist_train = ballist_train[:, 2:4]

x_ballist_test = ballist_test[:, 0:2]
y_ballist_test = ballist_test[:, 2:4]

RBF_net = RBF(2, 13, 2, 0, (2*np.pi))
RBF_net.train()

plt.scatter(x_ballist_train[:, 0], x_ballist_train[:, 1])
plt.xlim((-0.1, 1.1))
plt.ylim((-0.1, 1.1))
plt.show()

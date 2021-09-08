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

means_low = [0.0032970690764085263, 0.0025416024148464866, 0.0034569802222553523,
             0.007161932504873203, 0.010939013304975018, 0.005723280446585552]
means_mid = [0.012247284764049998, 0.008068369544183027, 0.004059740898614545,
             0.004482386411397175, 0.008811800132465786, 0.011588332733379999]
means_high = [0.016557924415187233, 0.0061268044959385815, 0.005527063442450035,
              0.005588563204114197, 0.01118537418879102, 0.01001461592380694]

reg_strengths = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
xi = list(range(len(reg_strengths)))

plt.plot(xi, means_low, label="0.03")
plt.plot(xi, means_mid, label="0.09")
plt.plot(xi, means_high, label="0.18")
""" plt.xlim((0, 8)) """
plt.xlabel("Regularization Strength")
plt.ylabel("Validation MSE Loss")
plt.xticks(xi, reg_strengths)
plt.legend(loc='upper right', title="Std. dev. of gaussian noise")
plt.show()

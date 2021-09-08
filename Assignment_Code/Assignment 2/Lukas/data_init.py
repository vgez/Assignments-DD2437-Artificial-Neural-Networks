import numpy as np

# input data
x = np.arange(0, (2*np.pi), 0.1)
x_test = np.arange(0.05, (2*np.pi), 0.1)
# target data, sin(2x)
t_sin = np.array(np.sin(2*x))
t_sin_test = np.array(np.sin(2*x_test))
# target data, square(2x)
t_sq = np.where(t_sin >= 0, 1, -1)
t_sq_test = np.where(t_sin_test >= 0, 1, -1)

print(len(x))
print(len(x_test))
print(len(t_sin))
print(len(t_sin_test))

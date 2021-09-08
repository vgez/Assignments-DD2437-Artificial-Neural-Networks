import numpy as np

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1]).reshape(1, -1)

y = np.matmul(x1.T, x1)
y2 = np.dot(x1.T, x1)
y3 = np.multiply(x1.T, x1)


compare_1_2 = y == y2
compare_1_3 = y == y3
compare_2_3 = y2 == y3

""" print(compare_1_2.all())
print(compare_1_3.all())
print(compare_2_3.all()) """


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


hej = check_symmetric(y)
print(hej)

print(y)

print(np.subtract(y, np.identity(8, dtype=int)))

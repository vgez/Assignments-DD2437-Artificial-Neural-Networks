import numpy as np
def multivariatenormal(mean, cov, size=100):
    x = np.random.multivariate_normal(mean, cov, size).T
    return x

from multivariatenormal import multivariatenormal
import numpy as np
import random
from labels import makelabels
def create_data_sep(mean1 = [2,1], mean2 = [-1,-3], cov=[[0.5, 0], [0,0.5]], n=100):
    norm1 = multivariatenormal(mean1, cov, n)
    norm2 = multivariatenormal(mean2, cov, n)
    shuffledx = np.concatenate((norm1[0], norm2[0])), np.concatenate((norm1[1], norm2[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    T = makelabels(n,1) + makelabels(n,-1)
    random.Random(4).shuffle(T)
    return shuffledx, T

def create_data_non_sep(mean1=[2,1], mean2=[-1,-3], mean3=[-10, -10], cov=[[0.5, 0], [0,0.5]], n=100):
    norm1 = multivariatenormal(mean1, cov, n)
    norm2 = multivariatenormal(mean2, cov, n)
    norm3 = multivariatenormal(mean3, cov, n)
    shuffledx = np.concatenate((norm1[0], norm2[0], norm3[0])), np.concatenate((norm1[1], norm2[1], norm3[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    T = makelabels(n,1) + makelabels(n,-1) + makelabels(n,1)
    random.Random(4).shuffle(T)
    return shuffledx, T

def create_data_xor(mean1=[10,10], mean2=[0,0], mean3=[0, 10], mean4=[10,0], cov=[[0.10, 0], [0,0.5]], n=20):
    norm1 = multivariatenormal(mean1, cov, n)
    norm2 = multivariatenormal(mean2, cov, n)
    norm3 = multivariatenormal(mean3, cov, n)
    norm4 = multivariatenormal(mean4, cov, n)

    shuffledx = np.concatenate((norm1[0], norm2[0], norm3[0], norm4[0])), np.concatenate((norm1[1], norm2[1], norm3[1], norm4[1]))
    shuffledx = np.array(shuffledx)
    random.Random(4).shuffle(shuffledx[0])
    random.Random(4).shuffle(shuffledx[1])
    T = makelabels(n,-1) + makelabels(n,-1) + makelabels(n,1) + makelabels(n,1)
    random.Random(4).shuffle(T)
    return shuffledx, T

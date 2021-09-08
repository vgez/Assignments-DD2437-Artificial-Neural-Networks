import matplotlib.pyplot as plt
import numpy as np
from multivariatenormal import multivariatenormal;
import random
from labels import makelabels
from perceptron import *
from perceptron import perceptronsinglelayerbatch
from delta import *
from neuralnet import NeuralNet
from activations import step
import time
# Create labeled learning data
mean1 = [2,1]
mean2 = [-1,-3]
mean3 = [-10,-10]
cov = [[0.5, 0], [0,0.5]]
n = 100
norm1 = multivariatenormal(mean1, cov, n)
norm2 = multivariatenormal(mean2, cov, n)
norm3 = multivariatenormal(mean3, cov, n)
shuffledx = np.concatenate((norm1[0], norm2[0])), np.concatenate((norm1[1], norm2[1]))
# shuffledx = np.concatenate((norm1[0], norm2[0], norm3[0])), np.concatenate((norm1[1], norm2[1], norm3[1]))
shuffledx = np.array(shuffledx)
random.Random(4).shuffle(shuffledx[0])
random.Random(4).shuffle(shuffledx[1])
# T = makelabels(n,1) + makelabels(n,-1) + makelabels(n,1)
T = makelabels(n,1) + makelabels(n,-1)
random.Random(4).shuffle(T)
# shuffledx= np.append(shuffledx, [[1 for i in range(2*n)]], axis=0)


def createline(normal, linelen):
    line = [[] for i in range(2)]
    x = np.linspace(-5,5,100)
    y = -((normal[0]*x)/normal[1]+normal[2]/normal[1])
    line[0][:] = x
    line[1][:] = y
    return line

def plotall(points, linein):
    plt.figure()
    for i in range(200):
        # Plots all points
        if T[i] == 1:
            plt.plot(points[0][i],points[1][i], 'b+')
        else:
            plt.plot(points[0][i],points[1][i]
                    ,'ro', markerfacecolor='none')

    # Plots line
    plt.plot(linein[0],linein[1],'g-')
    plotlim = 5
    plt.xlim(-plotlim, plotlim)
    plt.ylim(-plotlim, plotlim)
    plt.gca().set_aspect('equal', adjustable='box')
#     plt.draw()

def plotpoints(points):
    for i in range(200):
        # Plots all points
        if T[i] == 1:
            plt.plot(points[0][i],points[1][i], 'b+')
        else:
            plt.plot(points[0][i],points[1][i]
                    ,'ro', markerfacecolor='none')
    plotlim = 15
    plt.xlim(-plotlim, plotlim)
    plt.ylim(-plotlim, plotlim)
    plt.gca().set_aspect('equal', adjustable='box')

def pauseplot(time):
    plt.pause(time)

def clearplot():
    plt.clf()


def runperceptronseq():
    w =[5,-1,-10]
    for epoch in range(100):
        for i in range(200):
            # Which learning model to use
            w = perceptronsinglelayerseq(shuffledx[:,i], T[i], w)
        linew = createline(w,10)
        plotall(shuffledx, linew)
        pauseplot(0.001)
        clearplot()

    plt.show()

def runperceptronbatch():
    w =[5,-1,-10]
    for epoch in range(100):
        w = perceptronsinglelayerbatch(shuffledx, T, w)
        linew = createline(w,10)
        plotall(shuffledx, linew)
        pauseplot(0.001)
        clearplot()

def rundeltaseq():
    w =[5,-1,-10]
    for epoch in range(100):
        for i in range(200):
            # Which learning model to use
            w = deltasinglelayerseq(shuffledx[:,i], T[i], w)
        linew = createline(w,10)
        plotall(shuffledx, linew)
        pauseplot(0.001)
        clearplot()

    plt.show()

def rundeltabatch():
    w = [1, -1, -5]
    for epoch in range(100):
        w = deltasinglelayerbatch(shuffledx, T, w)
        linew = createline(w,10)
        plotall(shuffledx, linew)
        pauseplot(0.001)
        clearplot()

def rundeltamultibatch(n_nodes_v, epochs):
    start = time.time()
    # Create weight matrices
    net = NeuralNet()
    n_layers = len(n_nodes_v)-1
    print("Layers = ",n_layers)
    for l in range(n_layers+1):
        net.add_layer(n_nodes_v[l])

    print(net)
    net.generate_weights()
    err = []
    val = []
    x = []
    net.set_inputs(0, shuffledx)
    net.set_y(0, shuffledx)
    net.forward_pass()
    net.backward_pass(T)
    step_v = np.vectorize(step)
    first_classification = T - (np.ceil(net.get_outputs())*2-1)
#     print(net.get_outputs())
    for epoch in range(epochs):
        net.forward_pass()
        net.backward_pass(T)
        net.update()
#         print(np.mean(np.square(net.get_error())))
        x.append(epoch)
        err.append(np.mean(np.square(net.get_error())))
        val.append(np.count_nonzero(T - (np.ceil(net.get_outputs())*2-1)))

#     print(first_classification)
#     print(T - (np.ceil(net.get_outputs())*2-1))
#     print(T)
#     print(net.get_outputs())
#     print(np.ceil(net.get_outputs())*2-1)
    print("Time elapsed: ", round(time.time()-start, 3)," seconds")
    plot_err_val(x, err, val)
    # plotpoints(shuffledx)
    show_plots()


def plot_err_val(x,err,val):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(x, err)
    ax1.set_title("Mean square error")
    ax2.plot(x, val)

def show_plots():
    plt.show()

# runperceptronseq()
# runperceptronbatch()
# rundeltaseq()
# rundeltabatch()
rundeltamultibatch([2,3,1],5000)

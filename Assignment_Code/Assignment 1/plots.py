from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def get_3D_plot(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, z, cmap='summer')

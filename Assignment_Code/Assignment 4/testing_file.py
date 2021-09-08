import numpy as np
from util import *
from matplotlib import pyplot as plt
from rbm import RestrictedBoltzmannMachine
from PIL import Image


def plot_recon_err():

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000)

    hidden_units = [200, 300, 400, 500]
    #hidden_units = [200]
    results = []
    iters = 1000

    ''' restricted boltzmann machine '''

    print("\nStarting a Restricted Boltzmann Machine..")
    for i in range(len(hidden_units)):
        print(i)
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                         ndim_hidden=hidden_units[i],
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=10
                                         )

        results.append(rbm.cd1(visible_trainset=train_imgs,
                               n_iterations=iters, calc_err=True))

    epochs = np.arange(len(results[0]))
    for i in range(len(results)):
        plt.plot(epochs, results[i], label=str(hidden_units[i]) + " units")
    plt.legend(loc="upper right")
    plt.xticks(np.arange(0, 11, step=1))
    plt.xlabel("Iterations")
    plt.ylabel("Average reconstruction error")
    plt.show()


def plot_batch_probabilities():

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000)

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
                                     )

    rbm.cd1(visible_trainset=train_imgs, n_iterations=10000, calc_err=False)

    batch_input = test_imgs[:rbm.batch_size]
    p_h_given_v, _ = rbm.get_h_given_v(batch_input)
    image = p_h_given_v * 255
    image = Image.fromarray(image.astype(np.uint8))
    image.save('batch_probs_20_batch.png', format='PNG')


def plot_hist():
    pass


def recall_digits():

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000)

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10
                                     )

    rbm.cd1(visible_trainset=train_imgs, n_iterations=10000, calc_err=False)

    for i in range(10):
        plt.imshow(np.reshape(test_imgs[i], (28, 28)), cmap='gray')
        plt.title("Test image " + str(i+1))
        plt.show()
        """ print(test_imgs[i].shape)
        image = test_imgs[i] * 255
        image = image.reshape((28, 28))
        image = Image.fromarray(image.astype(np.uint8))
        image.save('img_init_' + str(i), format='PNG') """

        recon = rbm.reconstruct_img(test_imgs[i])

        """ image = recon * 255
        image = image.reshape((28, 28))
        image = Image.fromarray(image.astype(np.uint8))
        image.save('img_recon_' + str(i), format='PNG') """
        plt.imshow(np.reshape(recon, (28, 28)), cmap='gray')
        plt.title("Reconstructed image " + str(i+1))
        plt.show()


if __name__ == "__main__":
    plot_recon_err()

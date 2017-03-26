import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], np.array([0.299, 0.587, 0.114], dtype=np.float32))


def load(path):
    img = plt.imread(path)
    gray = rgb2gray(img)
    return np.reshape(gray, [1, 28 * 28])


def show(image):
    plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
    plt.show()


def run():
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    # image = mnist.test.images[:1]
    path = '/home/rean/projects/git/neural-network/images/foo.png'
    # plt.imsave(path, np.reshape(image, [28, 28]), cmap='gray')
    image = load(path)
    show(image)

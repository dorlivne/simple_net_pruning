import tensorflow as tf
import random
from cifar.cifar.cifar import CIFAR10
import math
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 24

def get_second_dimension(tensor):
    return tensor.get_shape().as_list()[1]

def process_set_mnist(set_x):
    set_x = set_x.reshape(-1, 28, 28, 1)
    return set_x

def load_mnist(num_val: int, num_train=30000):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # subsample the data
        mask = range(num_train, num_train + num_val)
        x_val = x_train[mask]
        y_val = y_train[mask]
        mask = range(0, num_train)
        x_train = x_train[mask]
        y_train = y_train[mask]
        x_train = process_set_mnist(x_train)
        x_test = process_set_mnist(x_test)
        x_val = process_set_mnist(x_val)
        x_train /= 255
        x_test /= 255
        return x_train, y_train, x_test, y_test, x_val, y_val


def get_batch(batch_size, x_set, y_set):
    mask_for_output = random.sample(range(x_set.shape[0]), batch_size)
    return x_set[mask_for_output], y_set[mask_for_output]


def load_cifar_10():
    # Instantiate the dataset. If the dataset is not found in `dataset_root`,
    #  the first time it is automatically downloaded and extracted there.
    dataset = CIFAR10(dataset_root='./cifar10')
    # You can convert the CIFARSamples to ndarray. Images are possibly flattened
    #  and/or normalized to be centered on zero (i.e. in range [-0.5, 0.5])
    x_train, y_train = CIFAR10.to_ndarray(dataset.samples['train'])
    x_test, y_test = CIFAR10.to_ndarray(dataset.samples['test'])
    num_of_train = int(math.ceil(x_train.shape[0] * 0.8))
    x_val = x_train[num_of_train:]
    y_val = y_train[num_of_train:]
    x_train = x_train[:num_of_train]
    y_train = y_train[:num_of_train]
    #x_train = jitter_images(x_train, y_train)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test


def imshow_noax(img, normalize=False):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 55.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def jitter_images(x_train):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    PLOT = False
    jitter_images = np.zeros((x_train.shape[0], height, width, 3))
    with tf.Session():
        tensor = tf.placeholder(dtype=tf.float32, shape=None, name="data_augmenting")
        for i, image in enumerate(x_train):
            if PLOT:
                imshow_noax(image)
            reshaped_image = image.astype(np.float32)
            if PLOT:
                imshow_noax(reshaped_image)
            distorted_image = tf.random_crop(tensor, [height, width, 3]).eval(feed_dict={tensor: reshaped_image})
            if PLOT:
                imshow_noax(distorted_image)
            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(tensor).eval(feed_dict={tensor: distorted_image})
            if PLOT:
                imshow_noax(distorted_image)
             # Because these operations are not commutative, consider randomizing
            # the order their operation.
            distorted_image = tf.image.random_brightness(tensor, max_delta=63).eval(feed_dict={tensor: distorted_image})
            distorted_image = tf.image.random_contrast(tensor, lower=0.2, upper=1.8).eval(feed_dict={tensor: distorted_image})
            if PLOT:
                imshow_noax(distorted_image)
            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_standardization(tensor).eval(feed_dict={tensor: distorted_image})
            if PLOT:
                imshow_noax(float_image)
            np.reshape(float_image, (height, width, 3))
            jitter_images[i, :, :, :] = float_image
            if PLOT:
                imshow_noax(jitter_images[i,:,:,:])
    return jitter_images






# An MNIST loader.

import numpy as np
import gzip
import struct


def load_images(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    # Insert a column of 1s in the position 0 of X.
    # (“axis=1” stands for: “insert a column, not a row”)
    return np.insert(X, 0, 1, axis=1)


def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


# Deprecated, use one_hot_encode to recognize all digits
def encode_fives(y):
    # Convert all 5s to 1, and everything else to 0
    return (y == 5).astype(int)


def one_hot_encode(y):
    n_labels = y.shape[0]
    n_classes = 10
    encoded_y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = y[i]
        encoded_y[i][label] = 1
    return encoded_y


def standardize(training_set, test_set):
    average = np.average(training_set)
    standard_deviation = np.std(training_set)
    training_set_standardized = (training_set - average) / standard_deviation
    test_set_standardized = (test_set - average) / standard_deviation
    return training_set_standardized, test_set_standardized


# X_train/X_validation/X_test: 60K/5K/5K images
# Each image has 784 elements (28 * 28 pixels)
X_train_raw = load_images("mnist_data/train-images-idx3-ubyte.gz")
X_test_raw = load_images("mnist_data/t10k-images-idx3-ubyte.gz")
X_train, X_test_all = standardize(X_train_raw, X_test_raw)
X_validation, X_test = np.split(X_test_all, 2)

# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels("mnist_data/train-labels-idx1-ubyte.gz")
# Y_train: 60K labels, each consisting of 10 one-hot-encoded elements
Y_train = one_hot_encode(Y_train_unencoded)
# Y_validation/Y_test: 5K/5K labels, each a single digit from 0 to 9
Y_test_all = load_labels("mnist_data/t10k-labels-idx1-ubyte.gz")
Y_validation, Y_test = np.split(Y_test_all, 2)

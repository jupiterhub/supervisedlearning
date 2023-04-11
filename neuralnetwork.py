import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)


def loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]


def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


def sigmoid_gradient(sigmoid):
    return np.multiply(sigmoid, (1 - sigmoid))


def back(x, y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - y)) / x.shape[0]
    w1_gradient = np.matmul(prepend_bias(x).T, np.matmul(y_hat - y, w2[1:].T)
                            * sigmoid_gradient(h)) / x.shape[0]
    return w1_gradient, w2_gradient

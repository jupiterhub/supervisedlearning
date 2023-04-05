import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat


# originally from mnist.py. Moved here since we need to prepend_bias in both input and hidden layers
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


def softmax(logits):
    exponential = np.exp(logits)
    return exponential / np.sum(exponential, axis=1).reshape(-1, 1)

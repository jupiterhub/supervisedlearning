# A binary classifier that recognizes one of the digits in MNIST.

import numpy as np


# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Basically doing prediction but named forward as its
# performing Forward-Propagation
def forward(x, w):
    weighted_sum = np.matmul(x, w)
    return sigmoid(weighted_sum)


# Calling the predict() function
def classify(x, w):
    return np.round(forward(x, w))


# Computing Loss over using logistic regression
def loss(x, y, w):
    y_hat = forward(x, w)
    first_term = y * np.log(y_hat)
    second_term = (1 - y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


# calculating gradient
def gradient(x, y, w):
    return np.matmul(x.T, (forward(x, w) - y)) / x.shape[0]


# calling the training function for desired no. of iterations
def train(x, y, iterations, lr):
    w = np.zeros((x.shape[1], 1))
    for i in range(iterations):
        print('Iteration %4d => Loss: %.20f' % (i, loss(x, y, w)))
        w -= gradient(x, y, w) * lr
    return w


# Doing inference to test our model
def test(x, y, w):
    total_examples = x.shape[0]
    correct_results = np.sum(classify(x, w) == y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))


# Test it
import mnist as data

w = train(data.X_train, data.Y_train, iterations=100, lr=1e-5)
# use a different data for test to avoid overfitting
test(data.X_test, data.Y_test, w)

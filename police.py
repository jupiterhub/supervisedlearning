# A binary classifier.

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


# calling the training function for 10,000 iterations
def train(x, y, iterations, lr):
    w = np.zeros((x.shape[1], 1))
    for i in range(iterations):
        if (i % 2000 == 0 or i == 9999):
            print("Iteration %4d => Loss: %.20f" % (i, loss(x, y, w)))
        w -= gradient(x, y, w) * lr
    return w


# Doing inference to test our model
def test(x, y, w):
    total_examples = x.shape[0]
    correct_results = np.sum(classify(x, w) == y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))


# Prepare data
x1, x2, x3, y_truth = np.loadtxt("police.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y_truth.reshape(-1, 1)
weight = train(X, Y, iterations=10000, lr=0.001)

# Test it
test(X, Y, weight)

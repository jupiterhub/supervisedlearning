# A binary classifier that recognizes one of the digits in MNIST.
import numpy as np
import mnist as data


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
    y_hat = forward(x, w)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


# Computing Loss over using logistic regression
def loss(x, y, w):
    y_hat = forward(x, w)
    first_term = y * np.log(y_hat)
    second_term = (1 - y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


# calculating gradient
def gradient(x, y, w):
    return np.matmul(x.T, (forward(x, w) - y)) / x.shape[0]


# Printing results to the terminal screen
def report(iteration, x_train, y_train, x_test, y_test, w):
    matches = np.count_nonzero(classify(x_test, w) == y_test)
    n_test_examples = y_test.shape[0]
    matches = matches * 100.0 / n_test_examples
    training_loss = loss(x_train, y_train, w)
    if (iteration % 20 == 0) or iteration == 199:
        print("%d - Loss: %.20f, %.2f%%" % (iteration, training_loss, matches))


# calling the training function for desired no. of iterations
def train(x_train, y_train, x_test, y_test, iterations, lr):
    w = np.zeros((x_train.shape[1], y_train.shape[1]))
    for i in range(iterations):
        report(i, x_train, y_train, x_test, y_test, w)
        w -= gradient(x_train, y_train, w) * lr
    report(iterations, x_train, y_train, x_test, y_test, w)
    return w


# Doing inference to test our model
def test(x, y, w):
    total_examples = x.shape[0]
    correct_results = np.sum(classify(x, w) == y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))


# Test it
w = train(data.X_train, data.Y_train,
          data.X_test, data.Y_test,
          iterations=200, lr=1e-5)

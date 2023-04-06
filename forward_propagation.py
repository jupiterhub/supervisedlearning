import numpy as np
import mnist as data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(x, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(x), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat


# originally from mnist.py. Moved here since we need to prepend_bias in both input and hidden layers
def prepend_bias(x):
    return np.insert(x, 0, 1, axis=1)


def softmax(logits):
    exponential = np.exp(logits)
    return exponential / np.sum(exponential, axis=1).reshape(-1, 1)


def classify(x, w1, w2):
    y_hat = forward(x, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


# cross-entropy loss, a simpler formula compared to log-loss
# works better with SOFTMAX
def loss(y, y_hat):
    return -np.sum(y * np.log(y_hat)) / y.shape[0]


def report(iteration, x_train, y_train, x_test, y_test, w1, w2):
    y_hat = forward(x_train, w1, w2)
    training_loss = loss(y_train, y_hat)
    classifications = classify(x_test, w1, w2)
    accuracy = np.average(classifications == y_test) * 100.0
    print("Iteration: %5d, Loss: %.6f, Accuracy: %.2f%%" %
          (iteration, training_loss, accuracy))

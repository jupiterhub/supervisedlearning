import numpy as np
import mnist


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


def forward(x, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(x), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat, h


def back(x, y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - y)) / x.shape[0]
    w1_gradient = np.matmul(prepend_bias(x).T, np.matmul(y_hat - y, w2[1:].T)
                            * sigmoid_gradient(h)) / x.shape[0]
    return w1_gradient, w2_gradient


def classify(x, w1, w2):
    y_hat, _ = forward(x, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def initialize_weights(n_input_variables, n_hidden_nodes, n_classes):
    w1_rows = n_input_variables + 1
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)

    return (w1, w2)


def prepare_batches(x_train, y_train, batch_size):
    x_batches = []
    y_batches = []
    n_examples = x_train.shape[0]
    for batch in range(0, n_examples, batch_size):
        batch_end = batch + batch_size
        x_batches.append(x_train[batch:batch_end])
        y_batches.append(y_train[batch:batch_end])
    return x_batches, y_batches



def report(iteration, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat, _ = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classifications = classify(X_test, w1, w2)
    accuracy = np.average(classifications == Y_test) * 100.0
    print("Iteration: %5d, Loss: %.8f, Accuracy: %.2f%%" %
          (iteration, training_loss, accuracy))


def train(x_train, y_train, x_test, y_test, n_hidden_nodes,
          epochs, batch_size, lr, print_every=10):
    n_input_variables = x_train.shape[1]
    n_classes = y_train.shape[1]

    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    x_batches, y_batches = prepare_batches(x_train, y_train, batch_size)
    report(0, x_train, y_train, x_test, y_test, w1, w2)
    for epoch in range(epochs):
        for batch in range(len(x_batches)):
            y_hat, h = forward(x_batches[batch], w1, w2)
            w1_gradient, w2_gradient = back(x_batches[batch], y_batches[batch],
                                            y_hat, w2, h)
            w1 = w1 - (w1_gradient * lr)
            w2 = w2 - (w2_gradient * lr)
        if (epoch + 1) % print_every == 0:
            report(epoch + 1, x_train, y_train, x_test, y_test, w1, w2)
    return w1, w2

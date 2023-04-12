import numpy as np
import neuralnetwork as nn


# One hot encoding
def one_hot_encode(Y):
    n_labels = Y.shape[0]
    result = np.zeros((n_labels, 2))
    for i in range(n_labels):
        result[i][Y[i]] = 1
    return result


# Loading no-linear data
x1, x2, y = np.loadtxt('non_linearly_separable.txt', skiprows=1, unpack=True)

# Performing One-Hot Encoding and then training the classifier
X_train = X_test = np.column_stack((x1, x2))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encode(Y_train_unencoded)
w1, w2 = nn.train(X_train, Y_train,
                  X_test, Y_test,
                  n_hidden_nodes=10, iterations=100000, lr=0.3)
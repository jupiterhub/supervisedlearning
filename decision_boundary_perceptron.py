import perceptron
import numpy as np

# One hot encoding
def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

# Loading data, prepending bias to it and finally one hot encoding training set
x1, x2, y = np.loadtxt('linearly_separable.txt', skiprows=1, unpack=True)
X_train = X_test = perceptron.prepend_bias(np.column_stack((x1, x2)))
Y_train_unencoded = Y_test = y.astype(int).reshape(-1, 1)
Y_train = one_hot_encode(Y_train_unencoded)

# Caling the training function
w = perceptron.train(X_train, Y_train, X_test, Y_test,
                                  iterations=10000, lr=0.1)
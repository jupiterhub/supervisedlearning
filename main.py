import numpy as np


# computing predictions
def predict(x, w):
    return np.matmul(x, w)


def loss(x, y, w):
    return np.average((predict(x, w) - y) ** 2)


def gradient(x, y, w):
    return 2 * np.matmul(x.T, (predict(x, w) - y)) / x.shape[0]


def train(x, y, iterations, lr):
    w = np.zeros((x.shape[1], 1))  # match number of rows with zeros, 1 column
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, (loss(x, y, w))))
        print("\tWeight: ", w.T)
        w -= gradient(x, y, w) * lr
    return w


# load data  - X (Pollution, Healthcare, Water) Y Life
x1, x2, x3, y_truth = np.loadtxt("life-expectancy.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))  # np.ones(x1.size) is to add bias of 1s
Y = y_truth.reshape(-1, 1)  # make it into a matrix, since multiplying matrix 1d-array to matrix produce unpredictable results
weight = train(X, Y, iterations=1000, lr=0.0001)

# Bias, Pollution, Healthcare, Water - Water has the most impact then healthcare
print("\nWeights: %s" % weight.T)  # transpose to rows for readability
print("\nA few predictions:")

# Some predictions are off because the hyperplane-based model is too simple
# We can only use linear regression if the points are roughly aligned to begin with
# complex real-world dataset needs a non-straight shape instead of a straight hyperplane
for i in range(10):
    # index, prediction, ground truth
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], weight), Y[i]))
# Plot the reservations/pizzas dataset.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


def plot(x, y):
    sea.set()
    plt.plot(x, y, "bo")
    plt.xlabel("Reservations")
    plt.ylabel("Pizzas")
    x_edge, y_edge = 50, 50
    plt.axis([0, x_edge, 0, y_edge])
    # Plot line
    plt.plot([0, x_edge], [b, predict(y_edge, w, b)], linewidth=1.0, color="g")
    plt.show()


def predict(x, weight):
    prediction = np.matmul(x, weight)
    # print("\t Prediction", prediction)
    return prediction


def loss(x, y, weight, ):
    loss_before_avg = (predict(x, weight) - y) ** 2
    # print("\t loss before avg", loss_before_avg)
    return np.average(loss_before_avg)


def gradient(x, y, weight):
    # X.T means transposed (covert rows -> columns)
    return 2 * np.matmul(x.T, (predict(x, weight) - y)) / x.shape[0]


def train_linear(x, y, iterations, lr):
    weight, bias = 0, 0
    for i in range(iterations):
        current_loss = loss(x, y, weight, bias)
        # print("Iteration weight=%f %4d => Loss: %.6f" % (weight, i, current_loss))

        if loss(x, y, weight + lr, bias) < current_loss:
            weight += lr
        elif loss(x, y, weight - lr, bias) < current_loss:
            weight -= lr
        elif loss(x, y, weight, bias + lr) < current_loss:
            bias += lr
        elif loss(x, y, weight, bias - lr) < current_loss:
            bias -= lr
        else:
            return weight, bias
    raise Exception("Couldn't converge within %d iterations" % iterations)


def train(x, y, iterations, lr):
    weight = np.zeros((x.shape[1], 1))  # no of inputs x 1 column matrix
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(x, y, weight)))
        weight -= gradient(x, y, weight) * lr
    return weight


# Import the dataset
x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)  # load data

# insert 1s at the start as a 'bias'
X = np.column_stack((np.ones(x1.size), x1, x2, x3))

# X.shape  # => (30, 3)
Y = y.reshape(-1, 1)  # fit the column with as many rows
# Y.shape  # => (30, 1)

# Train the system
w = train(X, Y, iterations=50000, lr=0.001)
print("\nweight=", w)
# first weight = bias, 2nd reservation, temp, tourist. Tourist has a large weight, temp has tiny one
# Therefore, Pizza's are impacted most by tourist and least by temp changes

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(5):
    # predict the needed pizzas
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))

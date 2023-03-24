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


def predict(x, weight, bias):
    prediction = x * weight + bias
    # print("\t Prediction", prediction)
    return prediction


def loss(x, y, weight, bias):
    loss_before_avg = (predict(x, weight, bias) - y) ** 2
    # print("\t loss before avg", loss_before_avg)
    return np.average(loss_before_avg)


def gradient(x, y, weight):
    return 2 * np.average(x * (predict(x, weight, 0) - y))  # fix bias for now


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
    weight = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, weight, 0)))
        weight -= gradient(X, Y, weight) * lr
    return weight


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)  # load data

# Train the system
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % w)

# Predict needed pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# Plot it
plot(X, Y)

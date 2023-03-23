# Plot the reservations/pizzas dataset.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


def plot(x, y):
    sea.set()
    plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
    plt.xticks(fontsize=14)                                  # set x axis ticks
    plt.yticks(fontsize=14)                                  # set y axis ticks
    plt.xlabel("Reservations", fontsize=14)                  # set x axis label
    plt.ylabel("Pizzas", fontsize=14)                        # set y axis label
    plt.plot(x, y, "bo")                                     # plot data
    plt.show()


def predict(x, w, b):
    prediction = x * w + b
    # print("\t Prediction", prediction)
    return prediction


def loss(x, y, w, b):
    loss_before_avg = (predict(x, w, b) - y) ** 2
    # print("\t loss before avg", loss_before_avg)
    return np.average(loss_before_avg)


def train(x, y, iterations, lr):
    w = 0
    b = 0
    for i in range(iterations):
        current_loss = loss(x, y, w, b)
        print("Iteration w=%f %4d => Loss: %.6f" % (w, i, current_loss))

        if loss(x, y, w + lr, b) < current_loss:
            w += lr
        elif loss(x, y, w - lr, b) < current_loss:
            w -= lr
        elif loss(x, y, w, b + lr) < current_loss:
            b += lr
        elif loss(x, y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)  # load data

# Train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# plot it
# plot(X, Y)
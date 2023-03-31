# Plots only the selected digit
import mnist
import numpy as np
import matplotlib.pyplot as plt

DIGIT = 5

X = mnist.load_images("mnist_data/train-images-idx3-ubyte.gz")
Y = mnist.load_labels("mnist_data/train-labels-idx1-ubyte.gz").flatten()
digits = X[Y == DIGIT]
np.random.shuffle(digits)

rows, columns = 3, 15
fig = plt.figure()
for i in range(rows * columns):
    ax = fig.add_subplot(rows, columns, i + 1)
    ax.axis('off')
    ax.imshow(digits[i].reshape((28, 28)), cmap="Greys")
plt.show()
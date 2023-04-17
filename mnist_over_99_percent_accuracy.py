import neuralnetwork as nn
import mnist as data

nn.train(data.X_train, data.Y_train, data.X_test, data.Y_test,
         n_hidden_nodes=1200, epochs=100, batch_size=600, lr=0.6)

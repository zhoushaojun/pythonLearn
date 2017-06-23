import numpy as np
from sklearn.datasets import load_digits


class Network(object):
    def __init__(self, sizes):
        self.num_lays = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoidw(np.dot(w, a) + b)
        return a

    def tostring(self):
        print(self.biases)
        print(self.weights)

    def wb(self):
        [print(w, b) for b, w in zip(self.biases, self.weights)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        print(zs)
        print(activations)
        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        print(delta)
        print(activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

network = Network([2,3, 1])
# network.tostring()
# print("--")
network.backprop([[1.0],[2.0]],[1.0])
#digits = load_digits()
#print(network.feedforward(digits.data[0, :].reshape(64, 1)))
# print(network.feedforward(digits.data[0]).shape)
# print(np.random.randn(2,2))

# print(np.random.normal(size=(2,2)))

print()

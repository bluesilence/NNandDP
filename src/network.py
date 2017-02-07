"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.
        For example, if the list was [2, 3, 1] then it would be a three-layer network,
        with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(num_neurons, 1) for num_neurons in sizes[1:]]
        self.weights = [np.random.randn(num_neurons_current_layer, num_neurons_previous_layer)
                        for num_neurons_previous_layer, num_neurons_current_layer
                        in list(zip(sizes[:-1], sizes[1:]))]

    def feedforward(self, activation):
        """Return the output of the network if ``activation`` is input."""
        for w, b in list(zip(self.weights, self.biases)):
            activation = sigmoid(np.dot(w, activation) + b)

        return activation

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data = None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        list_training_data = list(training_data)
        list_test_data = list(test_data)

        if test_data:
            n_test = len(list_test_data)
        
        n = len(list_training_data)

        for epoch in range(epochs):
            random.shuffle(list_training_data)
            mini_batches = [
                               list_training_data[k : min(k + mini_batch_size, n - 1)]
                               for k in range(0, n, mini_batch_size)
                           ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, mini_batch_size, eta)

            if test_data:
                print ('Epoch {0}: {1} / {2}'.format(epoch, self.evaluate(list_test_data), n_test))
            else:
                print ('Epoch {0} completed.'.format(epoch))

    def update_mini_batch(self, mini_batch, mini_batch_size, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays,
        similar to ``self.biases`` and ``self.weights``.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Backprop for each training data point (x, y) in the mini_batch
        for x, t in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, t)

            nabla_w = [nw + dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
            nabla_b = [nb + dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]

        # Update weights and biases after the mini_batch is done
        self.weights = [w - (eta / mini_batch_size) * nw
                        for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b - (eta / mini_batch_size) * nb
                        for b, nb in list(zip(self.biases, nabla_b))]

    def backprop(self, x, t):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays,
        similar to ``self.biases`` and ``self.weights``."""

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Feedforward
        activation = x
        activations = [ x ] # List to store all the activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer

        for w, b in list(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], t)

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.
        # Here, l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        # It's a renumbering of the scheme in the book,
        # used here to take advantage of the fact that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp

            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_b[-l] = delta

        return (nabla_w, nabla_b)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(predict == y) for (predict, y) in test_results)

    def cost_derivative(self, y, t):
        """Return the vector of partial derivatives of the cost function
        partial: {1/2 * (y - t)^2} / partial {y}
        for the output activations."""

        return (y - t)

import numpy as np

def sigmoid(x):
    # sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, inputs):
        return sigmoid(np.dot(np.array(inputs), self.weights))

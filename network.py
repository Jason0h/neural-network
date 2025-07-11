import numpy as np

from neuron import Neuron

def mse_loss(y_true, y_pred):
    return ((np.array(y_true) - np.array(y_pred)) ** 2).mean()

class NeuralNetwork:
    '''
    A neural network with:
        - 2 inputs (x1, x2)
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    '''

    def __init__(self):
        weights = [0, 1]
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
        
    def feedforward(self, x):
        h1_out = self.h1.feedforward(x)
        h2_out = self.h2.feedforward(x)
        o1_out = self.o1.feedforward([h1_out, h2_out])
        return o1_out
    
x = [2, 3]
network = NeuralNetwork()
print(network.feedforward(x))
print(mse_loss([1, -1, 1], [0.5, 0, 0.5]))
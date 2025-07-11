import numpy as np

from neuron import Neuron

def sigmoid(x):
    # sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # derivative of sigmoid activation function: f'(x) = f(x) * (1 - f(x))
    return sigmoid(x) * (1 - sigmoid(x))

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
        self.h0 = Neuron([np.random.normal(), np.random.normal()], np.random.normal())
        self.h1 = Neuron([np.random.normal(), np.random.normal()], np.random.normal())
        self.o0 = Neuron([np.random.normal(), np.random.normal()], np.random.normal())
        
    def feedforward(self, x):
        h0_out = self.h0.feedforward(x)
        h1_out = self.h1.feedforward(x)
        o0_out = self.o0.feedforward([h0_out, h1_out])
        return o0_out
    
    def predict(self, x):
        return 1 if self.feedforward(x) >= 0.5 else 0
    
    def train(self, data, all_y_trues):
        LEARN_RATE = 0.1
        EPOCHS = 1000
        for epoch in range(EPOCHS):
            for x, y_true in zip(data, all_y_trues):
                # calculate intermediate feedforward values
                sum_h0 = self.h0.weights[0] * x[0] + self.h0.weights[1] * x[1] + self.h0.bias
                h0 = sigmoid(sum_h0)

                sum_h1 = self.h1.weights[0] * x[0] + self.h1.weights[1] * x[1] + self.h1.bias
                h1 = sigmoid(sum_h1)

                sum_o0 = self.o0.weights[0] * h0 + self.o0.weights[1] * h1 + self.o0.bias
                o0 = sigmoid(sum_o0)
                y_pred = o0
                
                # calculate intermediate partial derivatives
                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_h0 = self.o0.weights[0] * deriv_sigmoid(sum_o0)
                d_ypred_d_h1 = self.o0.weights[1] * deriv_sigmoid(sum_o0)
                d_ypred_d_o0 = 1

                d_h0_d_w0 = x[0] * deriv_sigmoid(sum_h0)
                d_h0_d_w1 = x[1] * deriv_sigmoid(sum_h0)
                d_h0_d_b0 = deriv_sigmoid(sum_h0)
                d_h1_d_w2 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                d_o0_d_w4 = h0 * deriv_sigmoid(sum_o0)
                d_o0_d_w5 = h1 * deriv_sigmoid(sum_o0)
                d_o0_d_b2 = deriv_sigmoid(sum_o0)

                # update weights & biases (i.e. gradient descent step)
                self.h0.weights[0] -= LEARN_RATE * d_L_d_ypred * d_ypred_d_h0 * d_h0_d_w0
                self.h0.weights[1] -= LEARN_RATE * d_L_d_ypred * d_ypred_d_h0 * d_h0_d_w1
                self.h0.bias -= LEARN_RATE * d_L_d_ypred * d_ypred_d_h0 * d_h0_d_b0

                self.h1.weights[0] -= LEARN_RATE * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.h1.weights[1] -= LEARN_RATE * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.h1.bias -= LEARN_RATE * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.o0.weights[0] -= LEARN_RATE * d_L_d_ypred * d_ypred_d_o0 * d_o0_d_w4
                self.o0.weights[1] -= LEARN_RATE * d_L_d_ypred * d_ypred_d_o0 * d_o0_d_w5
                self.o0.bias -= LEARN_RATE * d_L_d_ypred * d_ypred_d_o0 * d_o0_d_b2

            # periodically calculate loss at the end of epoch
            if epoch % 100 == 0:
                y_preds = [self.feedforward(item) for item in data]
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %03d loss: %.3f" % (epoch, loss))
from network import *

data = [[-2, -1], [25, 6], [17, 4], [-15, -6]]
all_y_trues = [1, 0, 0, 1]

network = NeuralNetwork()
network.train(data, all_y_trues)

for item, ytrue in zip(data, all_y_trues):
    print(f"data {item} ypred {network.feedforward(item)} ytrue {ytrue}")
import numpy as np

# Multi Layer Perceptron
class mlp:
    
    # init architecture
    def __init__(self, dim, n_hidden, output):
        # input layer
        self.width = dim[0]
        self.height = dim[1]
        self.n_input = self.width * self.height + 1 # +1 for bias
        # hidden layer
        self.n_hidden = n_hidden + 1 # +1 for bias
        # output layer
        self.output = output
        self.n_output = len(output)
        # neuron
        self.input = np.zeros(self.n_input)
        self.hidden = np.zeros(self.n_hidden)
        self.output = np.zeros(self.n_hidden)
        # weight
        self.weight = [[],[]]
        # init weight 0 (input..hidden)
        self.weight[0] = np.random.rand(self.n_input*n_hidden)
        # init weight 1 (hidden..output)
        self.weight[1] = np.random.rand(n_hidden*self.n_output)
        # init bias = 1
        self.input[n_input-1] = 1
        self.hidden[n_hidden-1] = 1
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def activation(self, input, weight, b):
        z = np.dot(input, weight)
        return self.sigmoid(z)
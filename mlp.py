import numpy as np

# Multi Layer Perceptron
class mlp:

    # (5) AKTIVASI SIGMOID
    def activation(self, input, weight, b):
        z = np.dot(input, weight)
        return 1/(1 + np.exp(-z))

    def __init__(self, dim, n_hidden, output):
        # (2) DEFINISIKAN ARSITEKTUR
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
        
        # (6) BACK PROB : Init bobot dan bias
        # init weight [weight0, weight1]
        self.weight = [[],[]]
        # init weight 0 (input..hidden) = 0.5
        self.weight[0] = np.full(self.n_input*n_hidden, 0.5)
        # init weight 1 (hidden..output) = 0.5
        self.weight[1] = np.full(n_hidden*self.n_output, 0.5)
        # init bias(last index) = 1 
        self.input[self.n_input-1] = 1
        self.hidden[self.n_hidden-1] = 1

    # (7) BACK PROB : Hitung error
    def mse(self, predictions, targets):
        mse = 0
        n = len(predictions)
        for i in range(n):
            mse += (predictions[i] - targets[i])**2
        mse /= n
        return mse
        
    # (8) BACK PROB : Feed-forward
    def feed_forward(self, input, weight):
        return np.dot(input,weight)

    # (9) BACK PROB : Feed-backward
    def feed_backward(self, ):
        return 1

    # (10) BACK PROB : Prediksi
    # (11) BACK PROB : Measure akurasi
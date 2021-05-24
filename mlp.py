import numpy as np

# Multi Layer Perceptron
class mlp:

    # (5) AKTIVASI SIGMOID
    def activation(self, input, weight):
        z = np.dot(input, weight)
        return 1/(1 + np.exp(-z))

    def __init__(self, dim, n_hidden, output, epoch, alpha):
        # (2) DEFINISIKAN ARSITEKTUR
        # input layer
        self.width = dim[0]
        self.height = dim[1]
        self.n_input = self.width * self.height + 1 # +1 for bias
        self.input = np.zeros(self.n_input) # save input value
        
        # hidden layer
        self.hidden = np.zeros(self.n_hidden) # save activation value for hidden
        self.n_hidden = n_hidden + 1 # +1 for bias
        
        # output layer
        self.output = np.zeros(self.n_hidden) # save activation value for output
        self.n_output = len(output)
        
        
        # (6) BACK PROB : Init bobot dan bias
        # init weight [weight0, weight1]
        self.weight = [[],[]]
        # init weight 0 (input..hidden) = 0.5
        self.weight[0] = np.full((self.n_input, self.n_hidden), 0.5)
        # init weight 1 (hidden..output) = 0.5
        self.weight[1] = np.full((self.n_hidden, self.n_output), 0.5)
        # init bias(last index) = 1 
        self.input[self.n_input-1] = 1
        self.hidden[self.n_hidden-1] = 1

        # derivative
        self.derivative = np.zeros(self.n_hidden)

        # training log
        self.epoch = epoch
        self.alpha = alpha
        train_log = {'error' : [], 'accuracy' : []}

    # (7) BACK PROB : Hitung error
    def mse(self, predictions, targets):
        mse = 0
        n = len(predictions)
        for i in range(n):
            mse += (predictions[i] - targets[i])**2
        mse /= n
        return mse
        
    # (8) BACK PROB : Feed-forward
    def feed_forward(self):
        # update activation on hidden
        for i in range(self.n_hidden):
            self.hidden[i] = self.activation(self.input, self.weight[0][i])
        # update activation on output
        for i in range(self.n_output):
            self.output[i] = self.activation(self.hidden, self.weight[1][i])

    # (9) BACK PROB : Feed-backward
    def feed_backward(self, target):
        # update weight hidden..output
        self.derivative = np.zeros(self.n_hidden)
        for j in range(self.n_output):
            for i in range(self.n_hidden):
                a = self.hidden[i]
                derivative_ij = 2*(a-target[j])*a*(1-a) * self.weight[1][i][j]
                # update weight1 ij
                self.weight[1][i][j] -= derivative_ij
                # update derivative of neuron i on hidden layer
                self.derivative[i] += derivative_ij
        # update weight input..hidden
        for j in range(self.n_hidden):
            for i in range(self.n_input):
                # update weight0 ij
                self.weight[0][i][j] -= self.derivative[j]

    # (10) BACK PROB : Prediksi
    def predict(self, input):
        self.input = input
        self.feed_forward()
        max_idx = 0
        for i in range(self.n_output):
            if self.output[max_idx] < self.output[i]:
                max_idx = i #find maximum activation function
        if i == 0 :
            return [1,0,0]
        elif i == 1 :
            return [0,1,0]
        else :
            return [0,0,1]
          
    def train(self, datasets):
        for i in range(epoch):
import numpy as np
import sys
import time

# Multi Layer Perceptron
class mlp:

    # (5) AKTIVASI SIGMOID
    def activation(self, input, weight):
        z = np.dot(input, weight)
        return 1/(1 + np.exp(-z))

    def __init__(self, dim, n_hidden, targets, epoch, alpha):
        # (2) DEFINISIKAN ARSITEKTUR
        # input layer
        self.width = dim[0]
        self.height = dim[1]
        self.n_input = self.width * self.height + 1 # +1 for bias
        self.input = np.zeros(self.n_input) # save input value
        
        # hidden layer
        self.n_hidden = n_hidden + 1 # +1 for bias
        self.hidden = np.zeros(self.n_hidden) # save activation value for hidden
        
        # output layer
        self.n_output = len(targets)
        self.output = np.zeros(self.n_output) # save activation value for output
        
        # (6) BACK PROB : Init bobot dan bias
        # init weight [weight0, weight1]
        self.weight = [[],[]]
        # init weight 0 (input..hidden) = 0.5
        self.weight[0] = np.full((self.n_hidden, self.n_input), 0.5)
        # init weight 1 (hidden..output) = 0.5
        self.weight[1] = np.full((self.n_output, self.n_hidden), 0.5)
        # set bias(last index) = 1 
        self.input[self.n_input-1] = 1
        self.hidden[self.n_hidden-1] = 1

        # derivative
        self.derivative = np.zeros(self.n_hidden)

        # training log
        self.epoch = epoch
        self.alpha = alpha
        self.train_log = {'error' : [], 'accuracy' : []}

    # (7) BACK PROB : Hitung error
    def mse(self, activations, targets):
        mse = 0
        n = len(activations)
        for i in range(n):
            for j in range(self.n_output):
                mse += (activations[i][j] - targets[i][j])**2
        mse /= n*self.n_output
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
                derivative_ij = 2*(a-target[j])*a*(1-a) * self.weight[1][j][i]
                # update weight1 ij
                self.weight[1][j][i] -= self.alpha * derivative_ij
                # update derivative of neuron i on hidden layer
                self.derivative[i] += derivative_ij
        # update weight input..hidden
        for j in range(self.n_hidden):
            for i in range(self.n_input):
                # update weight0 ij
                self.weight[0][j][i] -= self.alpha * self.derivative[j]

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
        for e in range(self.epoch):
            start_time = time.perf_counter()
            print("training #"+str(e))
            activations = []
            targets = []
            n = len(datasets)
            for i in range(n):
                new_input = datasets[i][0].flatten()
                new_input = np.append(new_input, 1) # adding bias
                self.input = new_input
                self.feed_forward()
                self.feed_backward(datasets[i][1])
                activations.append(self.output)
                targets.append(datasets[i][1])
                # check performance
                done = i+1
                undone = n-done
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                remaining_time = undone/done*elapsed_time
                elapsed_time = "{:.2f}".format(elapsed_time)
                remaining_time = "{:.2f}".format(remaining_time)
                sys.stdout.write('\r'+"Progress : "+ str(done) + "/"+ str(n) + " - Elapsed = "+ str(elapsed_time) + "s - Remaining = " + str(remaining_time) + "s")
            mse = self.mse(activations, targets)
            print("Error =",mse)
            self.train_log['error'].append(mse)
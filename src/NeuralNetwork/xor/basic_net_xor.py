#!/usr/bin/env python2
import numpy as np
import math
import random

random.seed(1000)


class XorNeuralNet:

    def __init__(self,
                 input_nodes,
                 hidden_nodes,
                 output_nodes,
                 learning_rate,
                 activation='sigmoid'):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_deriv = self.sigmoid_deriv

        self.hidden_output = []
        self.all_nodes = (self.input_nodes + self.hidden_nodes) - 1
        self.weights = np.zeros((self.all_nodes, self.all_nodes))

    def initializeWeights(self):
        r = math.sqrt(6) / math.sqrt(self.input_nodes + self.hidden_nodes + 1)
        self.weights = np.random.random(
            (self.all_nodes, self.all_nodes)) * 2 * r - r

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def addBias(self, input_x):
        ones = np.atleast_2d(np.ones(input_x.shape[0]))
        input_x = np.concatenate((ones.T, input_x), axis=1)
        return input_x

    def feedForward(self, input_x):
        X = self.addBias(input_x)
        idx = np.random.randint(X.shape[0])
        randSampl = X[idx]
        print "All weights: \n", self.weights
        print "Total X:", randSampl
        # Feedforward hidden nodes
        for i in range(self.input_nodes):
            W_i = 0
            for j in range(self.input_nodes + 1):
                W_i += self.weights[j][i] * randSampl[j]
                print "H_%d :" % (i+1)
                print "Weight: ", self.weights[j][i], "index: ", "(", j, i, ")"
                print "X: ", randSampl[j]
            activation = self.sigmoid(W_i)
            self.hidden_output.append(activation)

        # Feedforward output nodes
        for i in range(self.output_nodes):
            W_i = 0.0
            for j in range(self.hidden_nodes):
                W_i += self.weights[j][i] * self.hidden_output[j]
            # This is for the b1 term
            W_i += self.weights[j+1][i] * randSampl[0]
            activation_h2 = self.sigmoid(W_i)
            self.hidden_output.append(activation_h2)

    def train(self, input_x, y, learning_rate=0.05, epochs=1):

        for k in range(epochs):
            if k % 100 == 0:
                print 'epochs: ', k
            self.feedForward(input_x)

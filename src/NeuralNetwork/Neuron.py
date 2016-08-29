#!/usr/bin/env python2
import math
import numpy as np


class Neuron(object):

    def __init__(self, input, inNumInputs):
        self.input = input
        self.numInputs = inNumInputs
        self.prevWeightDelta = []
        self.weights = []

    def initializeNeuronWeights(self):
        r = math.sqrt(6) / math.sqrt(10 + 10 + 1)
        for i in range(0, self.numInputs):
            self.weights.append(np.random.random() * 2 * r - r)
            self.prevWeightDelta.append(0.0)

    def setInput(self, input):
        self.input = input

    def getOutput(self):
        return self.output

    def setError(self, inError):
        self.error = inError

    def getError(self):
        return self.error

    def getWeights(self):
        return self.weights

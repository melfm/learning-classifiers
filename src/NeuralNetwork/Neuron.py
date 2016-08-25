#!/usr/bin/env python2
import math
import numpy as np


class Neuron(object):

    def __init__(self,
                 input,
                 prevWeightDelta,
                 numInputs):
        self.input = input
        self.numInputs = numInputs
        self.prevWeightDelta = prevWeightDelta

    def initializeNeuron(self, inInput, inNumInputs):
        self.input = inInput
        self.numInputs = inNumInputs
        self.weights = []
        self.prevWeightsDelta = []

        r = math.sqrt(6) / math.sqrt(10 + 10 + 1)
        for i in range(0, self.numInputs):
            self.weights.append(np.random.random() * 2 * r - r)
            self.prevWeightDelta.append(0.0)

    def setInput(input):
        self.input = input

    def getOutput(self):
        return self.output

    def setError(inError):
        self.error = inError

    def getError(self):
        return self.error

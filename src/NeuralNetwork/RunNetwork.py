#!/usr/bin/env python2
from Neuron import Neuron

if __name__ == '__main__':
    input = [1, 2, 3]
    neuron = Neuron(input, [], 3)

    neuron.initializeNeuron(input, 3)
    weights = neuron.getWeights()
    print weights


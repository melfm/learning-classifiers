#!/usr/bin/env python2
from Neuron import Neuron


def initializeNeuronWeights():
    input = [1, 2, 3]
    neuron = Neuron(input, len(input))
    neuron.initializeNeuronWeights()
    print neuron.weights


def setNeuronInput():
    input = [1, 2, 3]
    neuron = Neuron(input, len(input))
    anotherInput = [3, 4, 5]
    neuron.setInput(anotherInput)
    print neuron.input


def getNeuronWeight():
    input = [1, 2, 3]
    neuron = Neuron(input, len(input))
    neuron.initializeNeuronWeights()
    weights = neuron.getWeights()
    print weights

if __name__ == '__main__':
    initializeNeuronWeights()
    setNeuronInput()
    getNeuronWeight()

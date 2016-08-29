#! /usr/bin/env/ python2
from Neuron import Neuron

LEARNING_RATE = 0.2
ALPHA = 0.2


class Layer(object):

    def __init__(self, inNumNeurons):
        self.inNumNeurons = inNumNeurons

    def initializeLayer(self,
                        isInOutput,
                        isInInput,
                        inNextLayer=None,
                        inPrevLayer=None):
        self.isInOutput = isInOutput
        self.isInInput = isInInput
        self.nextLayer = inNextLayer
        self.prevLayer = inPrevLayer
        self.input = []
        self.output = []
        self.neurons = []
        self.numInputs = 0

    def connectInput(self, inInput, inNumInputs):
        self.numInputs = inNumInputs
        self.neurons = [Neuron(self.input, self.numInputs)
                        for i in range(self.inNumNeurons)]
        self.input = inInput

        for i in range(0, self.inNumNeurons):
            self.neurons[i].initializeNeuronWeights()

    def setInput(self, inInput):
        self.input = inInput
        for i in range(0, self.inNumNeurons -1):
            print self.inNumNeurons
            self.neurons[i].setInput(self.input)

    def feedForward():
        pass

    def backPropagate():
        pass

    def updateWeights():
        pass

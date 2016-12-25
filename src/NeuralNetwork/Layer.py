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
            print(self.inNumNeurons)
            self.neurons[i].setInput(self.input)

    def feedForward(self):
        for i in range(0, self.inNumNeurons):
            self.outputs[i] = self.neurons[i].fire()

        if(not self.isOutput):
            self.nextLayer.feedForward()

    def backPropagate(self):
        for i in range(0, self.inNumNeurons):
            deltaSum = 0
            for j in range(0, self.nextLayer.getNumNeurons()):
                layer_neuron = self.nextLayer.getNeuron[j]
                err = layer_neuron.getError()
                layer_weight = layer_neuron.getWeights()
                deltaSum += err * layer_weight
            self.neurons[i].setError(Neuron.sigmoid(self.neurons[i].getOutput()) * deltaSum)

    def updateWeights():
        pass

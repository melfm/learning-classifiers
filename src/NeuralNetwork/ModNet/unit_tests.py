#!/usr/bin/env python2
from Neuron import Neuron
from Layer import Layer
import unittest

import pdb

class NeuronTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        input = [1, 2, 3]
        cls.neuron = Neuron(input, len(input))

    def testInitializeNeuronWeights(self):
        input = [1, 2, 3]
        neuron = Neuron(input, len(input))
        neuron.initializeNeuronWeights()
        print(neuron.weights)


    def testsetNeuronInput(self):
        input = [1, 2, 3]
        neuron = Neuron(input, len(input))
        anotherInput = [3, 4, 5]
        neuron.setInput(anotherInput)
        print(neuron.input)


    def testgetNeuronWeight(self):
        print("Neuron fire test")
        input = [1, 2, 3]
        neuron = Neuron(input, len(input))
        neuron.initializeNeuronWeights()
        weights = neuron.getWeights()
        print(weights)

    def testNeuonFire(self):
        input = [1, 2, 3]
        neuron = Neuron(input, len(input))
        neuron.initializeNeuronWeights()
        output = neuron.fire()
        print(output)


class LayerTest(unittest.TestCase):

    def testinitializeLayer(self):
        layer_0 = Layer(1)
        layer_0.initializeLayer(False, False)

        input = [1, 2, 3]
        layer_0.setInput(input)

        layer_0.connectInput(input, 3)
        weights = layer_0.neurons[0].getWeights()
        print(weights)

    def testConnectLayer(self):
        pass

class BackpropTest(unittest.TestCase):

    def testInitialization(self):
        pass

    def testFeedForward(self):
        layer_0 = Layer(1)
        layer_0.initializeLayer(True,False)
        input = [5,6,7]
        layer_0.setInput(input)
        layer_0.connectInput(input, 3)

        layer_0.feedForward();
        print("Output of Layer", layer_0.output)

    def testFeedForwardMultiLayer(self):
        layer_0 = Layer(1)
        layer_1 = Layer(1)
        layer_1.initializeLayer(True, False, inPrevLayer=layer_0)

        layer_0.initializeLayer(False,False, inNextLayer=layer_1)
        input = [5,6,7]
        layer_0.setInput(input)
        pdb.set_trace()
        layer_1.setInput(layer_0)
        layer_1.connectInput(layer_0, 1)
        layer_0.connectInput(input, 3)
        layer_0.feedForward();
        pdb.set_trace()
        print("Output of Layer", layer_0.output)


    def testBackprop(self):
        pass

if __name__ == '__main__':
    unittest.main()

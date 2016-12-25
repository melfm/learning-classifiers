#!/usr/bin/env python2
from Neuron import Neuron
from Layer import Layer
import unittest

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

class BackpropTest(unittest.TestCase):

    def testInitialization(self):
        pass

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python2
from Layer import Layer


def initializeLayer():
    layer_0 = Layer(1)
    layer_0.initializeLayer(False, False)

    input = [1, 2, 3]
    layer_0.setInput(input)

    layer_0.connectInput(input, 3)
    weights = layer_0.neurons[0].getWeights()
    print weights


if __name__ == '__main__':
    initializeLayer()

#!/usr/bin/env python2
from basic_net_xor import XorNeuralNet
import numpy as np

xor = XorNeuralNet(2, 2, 1, 0.05)

xor.initializeWeights()

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 1, 1, 0])

xor.train(X, y)


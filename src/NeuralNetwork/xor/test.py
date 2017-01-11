#!/usr/bin/env python2
from basic_net_xor import XorNeuralNet
import numpy as np
import unittest

import pdb


class TestXorNeuralNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        input_nodes = 2
        hidden_nodes = 2
        output_node = 1
        learning_rate = 0.05
        cls.xor = XorNeuralNet(input_nodes,
                               hidden_nodes,
                               output_node,
                               learning_rate)

        # Dummy initialize the weight
        cls.xor.weights = [[0.4786365, -0.14597397, 2.39775668],
                           [-0.2638455,  -0.06040556,  1.36659629],
                           [-0.25551341, -0.18868968,  0.19195486]]

    def test_addBias(self):
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        biased_input = self.xor.addBias(X)
        self.assertEqual(biased_input.shape[1],
                         X.shape[1] + 1,
                         "Input/Bias Array shape mismatch")

        all_ones = biased_input[:, 0]
        self.assertEquals(sum(all_ones), 4)

    def test_feedForward(self):
        sampl_X = [[1, 0, 1]]
        self.xor.feedForward(sampl_X, 0)

        expected_act_hn1 = 0.5555505
        expected_act_hn2 = 0.4171063
        expected_act_o1 = 0.89032311

        self.assertTrue(self.xor.neuron_activations)
        self.assertAlmostEqual(self.xor.neuron_activations[0],
                               expected_act_hn1,
                               msg="First neuron activation is wrong")
        self.assertAlmostEqual(self.xor.neuron_activations[1],
                               expected_act_hn2,
                               msg="Second neuron activation is wrong")
        self.assertAlmostEqual(self.xor.neuron_activations[2],
                               expected_act_o1,
                               msg="Output neuron activation is wrong")

    def test_backpropagation(self):
        pass


if __name__ == '__main__':
    unittest.main()

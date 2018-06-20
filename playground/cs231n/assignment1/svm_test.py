import numpy as np
import time
import unittest

from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized


class SVMTest(unittest.TestCase):

    dump_output = True

    @classmethod
    def setUpClass(cls):
        # Small test to help understand the SVM loss function better

        cls.W = np.array([[-0.8125, -0.0381],
                          [0.0708, 0.0094]])
        cls.X = np.array([[-105.64189796, -115.98173469],
                          [-50.64189796,  -58.98173469]])

        cls.y = np.array([1, 1])

        cls.reg = 0.000005

    def test_svm_loss_naive(self):

        loss, dW = svm_loss_naive(self.W, self.X, self.y, self.reg)
        if self.dump_output:
            print('Naive dummy loss', loss)
            print('Naive dummy weights', dW)

        self.assertAlmostEqual(loss, 56.14, places=2)

    def test_svm_loss_vectorized(self):

        loss, dW = svm_loss_vectorized(self.W, self.X, self.y, self.reg)
        if self.dump_output:
            print('Vect dummy loss', loss)
            print('Vect dummy weights', dW)

        self.assertAlmostEqual(loss, 56.14, places=2)

if __name__ == '__main__':
    unittest.main()


#!/usr/bin/env python2
import numpy as np
import pylab as pb


def LoadData(show_data=False):
    data = np.loadtxt('src/RegressionAnalysis/ex1data1.txt', delimiter=',')
    if show_data:
        # Plot the data
        pb.scatter(data[:, 0], data[:, 1], marker='o', c='b')
        pb.title('Profits distribution')
        pb.xlabel('Population of City in 10,000s')
        pb.ylabel('Profit in $10,000s')
        pb.show()
    return data

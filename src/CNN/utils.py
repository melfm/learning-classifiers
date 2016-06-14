'''
@note: utility functions such as handling datasets.
'''
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import random


def load_CIFAR_batch(filename):
	"""
	load single batch of cifar-10 dataset
	"""
	with open(filename, 'r') as f:
		datadict=pickle.load(f)

		X=datadict['data']
		Y=datadict['labels']
		X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
		Y=np.array(Y)
		return X, Y

def load_COFAR10(ROOT):

	xs=[]
	ys=[]

	for b in range(1,6):
		f=os.path.join(ROOT, "data_batch_%d" % (b, ))
		X, Y=load_CIFAR_batch(f)
		xs.append(X)
		ys.append(Y)

	Xtr=np.concatenate(xs)
	Ytr=np.concatenate(ys)

	del X, Y

	Xte, Yte=load_CIFAR_batch(os.path.join(ROOT, "test_batch"))

	return Xtr, Ytr, Xte, Yte	

def visualize_CIFAR(X_train,
					y_train,
					samples_per_class):
	"""
	A visualize function for CIFAR 
	"""
	
	classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
	num_classes=len(classes);
	
	for y, cls in enumerate(classes):
		idxs=np.flatnonzero(y_train == y)
		idxs=np.random.choice(idxs, samples_per_class, replace=False)
		for i, idx in enumerate(idxs):
			plt_idx = i * num_classes + y + 1
			plt.subplot(samples_per_class, num_classes, plt_idx)
			plt.imshow(X_train[idx].astype('uint8'))
			plt.axis('off')
			if i == 0:
				plt.title(cls)
	
	plt.show();

def loadIrisDataSet(filename, split, trainingSet=[], testSet=[]):
	dataDir = os.path.abspath(os.path.join(os.path.dirname("iris.data"), '..', 'sampleData/iris.data'))
	with open(dataDir, 'rb') as csvfile:
		lines=csv.reader(csvfile)
		dataset=list(lines)
		for x in range(len(dataset) - 1):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			# randomly split data into train and test
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])



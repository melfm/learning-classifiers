import numpy as np
import math
import operator

class KNearestNeighbor:
	"""
	@note: this class provides an implementation of NearestNeighbor classifier.
	"""
	def __init__(self):
		pass

	###################################################
	# Similarity
	###################################################
	def euclideanDistance(self, instance1, instance2, length):
		distance = 0
		for x in range(length):
			distance += pow( (instance1[x] - instance2[x]) , 2)
		return math.sqrt(distance)

	###################################################
	# Neighbors 
	###################################################
	def getNeighbors(self, trainingSet, testInstance, k):
		distances = []
		length = len(testInstance) - 1
		for x in range(len(trainingSet)):
			dist = self.euclideanDistance(trainingSet[x], testInstance, length)
			distances.append((trainingSet[x], dist))
		distances.sort(key=operator.itemgetter(1))

		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	###################################################
	# Response 
	# devise a predicted response based on neighbors
	###################################################
	def predict_labels(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else :
				classVotes[response] = 1

		sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse = True)
		return sortedVotes[0][0]

	def predict(self, trainingSet, testSet, k):
		# generate predictions
		predictions = []
		for x in range(len(testSet)):
			neighbors = self.getNeighbors(trainingSet, testSet[x], k)
			result = self.predict_labels(neighbors)
			predictions.append(result)
			print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))\

	###################################################
	# Accuracy 
	###################################################
	def getAccuracy(self, testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct += 1
		return (correct/float(len(testSet))) * 100.0



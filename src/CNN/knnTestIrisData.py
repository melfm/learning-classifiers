from kNearestNeighborClassifier import *
from utils import loadIrisDataSet


nn = KNearestNeighbor()

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = nn.euclideanDistance(data1, data2, 3)
print 'Distance: ' + repr(distance)

#########################################################

trainSet = [[2,2,2,'a'], [4,4,4,'b']]
testInstance = [5,5,5]
k = 1
neighbours = nn.getNeighbors(trainSet, testInstance, 1)
print neighbours
print neighbours [0][-1]

#########################################################


# prepare data
trainingSet=[]
testSet=[]
split = 0.67
loadIrisDataSet('iris.data', split, trainingSet, testSet)
print 'Train set: ' + repr(len(trainingSet))
print 'Test set: ' + repr(len(testSet))

# generate predictions
predictions=[]
k = 3
for x in range(len(testSet)):
	neighbors = nn.getNeighbors(trainingSet, testSet[x], k)
	result = nn.predict_labels(neighbors)
	predictions.append(result)
	#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = nn.getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

import numpy as np
import pandas as pd
import random
from decisiontree import Node

class DecisionTree:

    def __init__(self, maxDepth = None, maxFeatures = None, randomState = None):
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.randomState = randomState
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        if(self.maxFeatures == None):
            self.maxFeatures = self.n_features
        if isinstance(self.maxFeatures, float) and self.maxFeatures <= 1:
            self.maxFeatures = (int)(self.maxFeatures * self.maxFeatures)

        self.tree = self.growTree(X, y, self.randomState)

    def growTree(self, X, y, randomState, depth=0):
        numSamplesPerClass = [np.sum(y == i) for i in range(self.n_classes)]
        predictedClass = np.argmax(numSamplesPerClass)
        node = Node.Node(predicted_class=predictedClass)

        if (self.maxDepth is None) or (depth < self.maxDepth):
            id, thr = self.bestSplit(X, y, randomState)

            if id is not None:
                if randomState is not None:
                    randomState += 1

                indicesLeft = X[:, id] < thr
                XLeft, yLeft = X[indicesLeft], y[indicesLeft]
                XRight, yRight = X[~indicesLeft], y[~indicesLeft]

                node.featureIndex = id
                node.threshold = thr
                node.left = self.growTree(XLeft, yLeft, randomState, depth + 1)
                node.right = self.growTree(XRight, yRight, randomState, depth + 1)
        return node

    def bestSplit(self, X, y, randomState):
        m = len(y)
        if (m <= 1):
            return None, None
        numClassParent = [np.sum(y == c) for c in range(self.n_classes)]
        bestGini = 1.0 - sum((n / m) ** 2 for n in numClassParent)
        if bestGini == 0:
            return None, None
        bestFeatId, bestThreshold = None, None
        random.seed(randomState)
        featInd = random.sample(range(self.n_features), self.maxFeatures)

        for featID in featInd:
            sortedColumn = sorted(set(X[:, featID]))
            thresholdValues = [np.mean([a, b]) for a, b in zip(sortedColumn, sortedColumn[1:])]

            for threshold in thresholdValues:
                leftY = y[X[:, featID] < threshold]
                rightY = y[X[:, featID] > threshold]

                numClassLeft = [np.sum(leftY == c) for c in range(self.n_classes)]
                numClassRight = [np.sum(rightY == c) for c in range(self.n_classes)]

                giniLeft = 1.0 - sum((n / len(leftY)) ** 2 for n in numClassLeft)
                giniRight = 1.0 - sum((n / len(rightY)) ** 2 for n in numClassRight)

                gini = (len(leftY) / m) * giniLeft + (len(rightY) / m) * giniRight

                if (gini < bestGini):
                    bestGini = gini
                    bestFeatId = featID
                    bestThreshold = threshold

        return bestFeatId, bestThreshold

    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        predictedClasses = np.array([self.predictExample(inputs) for inputs in X])

        return predictedClasses

    def predictExample(self, inputs):

        node = self.tree

        while node.left:
            if inputs[node.featureIndex] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predictedClass

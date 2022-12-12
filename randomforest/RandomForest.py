
import numpy as np
import pandas as pd

from scipy import stats
from decisiontree import DecisionTree

class RandomForest:
    def __init__(self, numTrees = 5, subsampleSize=None, maxDepth=None, maxFeatures=None, bootstrap=True, randomState=None):
        self.numTrees = numTrees
        self.subsampleSize = subsampleSize
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.bootstrap = bootstrap
        self.randomState = randomState
        self.decisionTrees = []

    def fit(self, X, y):
        if len(self.decisionTrees) > 0:
            self.decisionTrees = []

        numBuilt = 0

        while numBuilt < self.numTrees:
            print("Tree num. ", numBuilt)
            clf = DecisionTree.DecisionTree(maxDepth=self.maxDepth,
                               maxFeatures=self.maxFeatures,
                               randomState=self.randomState
            )

            _X, _y = self.Sample(X, y, self.randomState)

            clf.fit(_X, _y)
            self.decisionTrees.append(clf)
            numBuilt += 1

            if self.randomState is not None:
                self.randomState += 1

    def Sample(self, X, y, randomState):
        nRows, nCols = X.shape
        if self.subsampleSize is None:
            sampleSize = nRows
        else:
            sampleSize = int(nRows * self.subsampleSize)

        np.random.seed(randomState)
        samples = np.random.choice(a=nRows, size=sampleSize, replace=self.bootstrap)

        return X[samples], y[samples]

    def predict(self, X):
        y = []
        for tree in self.decisionTrees:
            y.append(tree.predict(X))

        y = np.swapaxes(y, axis1=0, axis2=1)

        predictedClasses = stats.mode(y, axis=1)[0].reshape(-1)
        return predictedClasses
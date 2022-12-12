import numpy as np
import pandas as pd
from randomforest import RandomForest

print("====Loading has started====")
testSet = pd.read_csv("rsc/mnist_test.csv")
testSet = np.array(testSet)
yTest = testSet[0:50000, 0]
XTest = testSet[0:50000, 1:]
yCross = testSet[50000:, 0]
XCross = testSet[50000:, 1:]

trainSet = pd.read_csv("rsc/mnist_train.csv")
trainSet = np.array(trainSet)
yTrain = trainSet[:, 0]
XTrain = trainSet[:, 1:]
print("Start")

numTrees = 1
subsampleSize = 0.001
maxDepth = 5
maxFeatures = 5
while numTrees < 100:
    while subsampleSize <= 1:
        while maxDepth < 100:
            while maxFeatures < 20:
                rf = RandomForest.RandomForest(numTrees=numTrees, subsampleSize=subsampleSize, maxDepth=maxDepth, maxFeatures=maxFeatures, bootstrap=True, randomState=1)
                rf.fit(XTrain, yTrain)

                results = rf.predict(X=XTest)
                count = 0
                for i in range (len(results)):
                    if results[i]==yTest[i]:
                        count+=1
                print(numTrees, " ", subsampleSize, " ", maxDepth, " ", maxFeatures, " ", 100*count/len(results))

                maxFeatures+=2

            maxDepth+=5

        subsampleSize*=10

    numTrees+=5


import numpy as np
import pandas as pd
import xlwt
from xlwt import Workbook
from randomforest import RandomForest


print("====Načítání datasetu zahájeno====")
#Načtení trénovací sady(60 000 číslic)
trainSet = pd.read_csv("rsc/mnist_train.csv")
trainSet = np.array(trainSet)
#Rozdělení sady na trénovací a cross validation v poměru 5:1
yTrain = trainSet[0:50000, 0]
XTrain = trainSet[0:50000, 1:]
yCrossValidation = trainSet[50000:, 0]
XCrossValidation = trainSet[50000:, 1:]

#Načtení testovací sady(10 000 číslic)
testSet = pd.read_csv("rsc/mnist_test.csv")
testSet = np.array(testSet)
yTest = testSet[:, 0]
XTest = testSet[:, 1:]
print("====Dataset byl načten====")

print("====Učení náhodných lesů zahájeno====")

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
sheet1.write(0, 0, 'Počet stromů')
sheet1.write(0, 1, 'Velikost vzorku')
sheet1.write(0, 2, 'Maximální hloubka')
sheet1.write(0, 3, 'Maximální počet vlastností')
sheet1.write(0, 4, 'Přesnost')
row = 1
numTrees = 1
while numTrees <= 100:
    subsampleSize = 0.1
    while subsampleSize < 1:
        maxDepth = 5
        while maxDepth < 100:
            maxFeatures = 5
            while maxFeatures <= 100:
                rf = RandomForest.RandomForest(numTrees=numTrees, subsampleSize=subsampleSize, maxDepth=maxDepth,
                                               maxFeatures=maxFeatures, bootstrap=False, randomState=1)
                rf.fit(XTrain, yTrain)

                results = rf.predict(X=XCrossValidation)
                count = 0
                for i in range(len(results)):
                    if results[i] == yCrossValidation[i]:
                        count += 1
                print(numTrees, " ", subsampleSize, " ", maxDepth, " ", maxFeatures, " ", 100*count/len(results), " %")
                #Zápis do excel tabulky
                sheet1.write(row, 0, numTrees)
                sheet1.write(row, 1, subsampleSize)
                sheet1.write(row, 2, maxDepth)
                sheet1.write(row, 3, maxFeatures)
                sheet1.write(row, 4, 100*count/len(results))
                row += 1

                maxFeatures = maxFeatures + 5

            maxDepth = maxDepth + 5

        subsampleSize = subsampleSize * 10

    numTrees = numTrees + 5

print("====Učení náhodných lesů ukončeno====")
wb.save('RndForest.xls')


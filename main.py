from builtins import len


from xlwt import Workbook
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from randomforest import RandomForest
from keras.utils import to_categorical
import keras
from keras import Input
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.activation import LeakyReLU

model = "CNN"

print("====Načítání datasetu zahájeno====")
#Načtení trénovací sady(60 000 číslic)
trainSet = pd.read_csv("rsc/mnist_train.csv")
trainSet = np.array(trainSet)
y = trainSet[:, 0]
X = trainSet[:, 1:]
X = X/255.0
X = scale(X)
#Rozdělení sady na trénovací a cross validation v poměru 5:1
yTrain = y[0:50000]
XTrain = X[0:50000, :]
yCrossValidation = y[50000:]
XCrossValidation = X[50000:, :]
#Načtení testovací sady(10 000 číslic)
testSet = pd.read_csv("rsc/mnist_test.csv")
testSet = np.array(testSet)
yTest = testSet[:, 0]
XTest = testSet[:, 1:]
XTest = XTest/255.0
print("====Dataset byl načten====")


if model == "RF":
    print("====Učení náhodných lesů zahájeno====")
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Počet stromů')
    sheet1.write(0, 1, 'Velikost vzorku')
    sheet1.write(0, 2, 'Maximální hloubka')
    sheet1.write(0, 3, 'Maximální počet vlastností')
    sheet1.write(0, 4, 'Přesnost')
    row = 1

    maxFeatures = 320
    maxDepth = 20
    subsampleSize = 0.1
    numsTrees = [100]

    for numTrees in numsTrees:
        rf = RandomForest.RandomForest(numTrees=numTrees, subsampleSize=subsampleSize, maxDepth=maxDepth,
                                          maxFeatures=maxFeatures, bootstrap=True, randomState=1)
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
        wb.save('RndForest.xls')

    print("====Učení náhodných lesů ukončeno====")

elif model == "SVM":
    hyperParameterGamma = [0.00001, 0.00005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'C')
    sheet1.write(0, 1, 'Gama')
    sheet1.write(0, 2, 'Přesnost')
    row = 1
    print("====Učení Support Vector Machine zahájeno====")
    for gamma in hyperParameterGamma:
        non_linear_model = SVC(kernel='rbf', C=9, gamma=0.01)
        non_linear_model.fit(XTrain, yTrain)
        results = non_linear_model.predict(XCrossValidation)
        count = 0
        for i in range(len(results)):
            if results[i] == yCrossValidation[i]:
                count += 1
        sheet1.write(row, 0, 9)
        sheet1.write(row, 1, gamma)
        sheet1.write(row, 2, 100 * count / len(results))
        row += 1
        wb.save('SVM.xls')
    print("====Učení Support Vector Machine ukončeno====")

elif model == "CNN":
    XTrain.shape = (50000, 28, 28, 1)
    XCrossValidation.shape = (10000, 28, 28, 1)
    XTest.shape = (10000, 28, 28, 1)

    yTrainCategorical = to_categorical(yTrain)
    yCrossValidationCategorical = to_categorical(yTrain)
    yTestCategorical = to_categorical(yTrain)

    batchSize = 64
    epochs = 20
    numClasses = 10
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28,28,1), padding='same'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D((2,2), padding='same'))
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='linear'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Dense(numClasses, activation='softmax'))
    cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnnTrain = cnn.fit(XTrain, yTrain, batch_size=batchSize, epochs=20, verbose=1, validation_data=(XCrossValidation, yCrossValidation))


import numpy as np


class NaiveBayesianClassifer:

    def __init__(self):
        super().__init__()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.loadData()
        self.mean=self.getMean(self.x_train)
        self.std=self.getStd(self.x_train)
        self.y= np.count_nonzero(self.y_train==1)/self.y_train.size

    def loadData(self):
        testSet = np.genfromtxt("./data/test.csv", delimiter=",")
        trainSet = np.genfromtxt("./data/train.csv", delimiter=",")
        x_train = trainSet[:, :8]
        y_train = trainSet[:, 8]
        x_test = testSet[:, :8]
        y_test = testSet[:, 8]
        return (x_train, y_train), (x_test, y_test)

    def getMean(self, input):
        shape = np.shape(input)
        if len(shape) == 1:
            result = np.ones(1)
        else:
            result = np.ones(shape[len(shape) - 1])

        block = 1 if len(shape) == 1 else shape[len(shape) - 1]

        for i in range(block):
            result[i] = np.var(input.flatten()[i::block])

        return result

    def getStd(self, input):
        shape = np.shape(input)
        if len(shape) == 1:
            result = np.ones(1)
        else:
            result = np.ones(shape[len(shape) - 1])

        block = 1 if len(shape) == 1 else shape[len(shape) - 1]

        for i in range(block):
            result[i] = np.std(input.flatten()[i::block])

        return result


    def getLikelihood(self, input):
        print(input)
            

    def classify(self):
        print("Classifying...")

    def calculateAccuracy(self):
        print("Calculating accuracy...")


nbc = NaiveBayesianClassifer()
print(nbc.y)
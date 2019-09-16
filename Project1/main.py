import numpy as np


class NaiveBayesianClassifer:
    def __init__(self):
        super().__init__()
        print(self.loadData())

    """
    Load data into program. 
    """
    def loadData(self):
        print("Loading data...")
        testSet = np.genfromtxt("./data/test.csv", delimiter=",")
        trainSet = np.genfromtxt("./data/train.csv", delimiter=",")
        x_train = trainSet[:, :8]
        y_train = trainSet[:, 8]
        x_test = testSet[:, :8]
        y_test = testSet[:, 8]
        return (x_train, y_train), (x_test, y_test)

    def calculatesMean(self):
        print("Calculating mean...")

    def calculateStandardDeviation(self):
        print("Calculating standard devication...")

    def calculateNormalDistributionLikelihood(self):
        print("Calculating Normal Distribution Likelihood...")

    def classify(self):
        print("Classifying...")

    def calculateAccuracy(self):
        print("Calculating accuracy...")


nbc = NaiveBayesianClassifer()
nbc.loadData()

import numpy as np
from sklearn.naive_bayes import GaussianNB


class NaiveBayesianClassifer:
    def __init__(self):
        super().__init__()

        #Load data
        self.train, self.test = self.loadData()

        #Dimension of X
        numFeatures = 8 

        #Calculate universal means, variances, and prior probability
        self.mean = self.getMean(self.train)[:numFeatures]
        self.var = self.getVar(self.train)[:numFeatures]
        self.pY = [x[numFeatures] for x in self.train].count(1) / self.train.size

        #Calculate means and variances with respect to y=1
        self.meanPY = self.getMean(self.train[self.train[:, numFeatures] == 1])[:numFeatures]
        self.varPY = self.getVar(self.train[self.train[:, numFeatures] == 1])[:numFeatures]

        #Calculate means and variance with respect to y=0
        self.meanPNotY = self.getMean(self.train[self.train[:, numFeatures] == 0])[:numFeatures]
        self.varPNotY = self.getVar(self.train[self.train[:, numFeatures] == 0])[:numFeatures]

        #Points to keep track of models performance
        self.TN = 0
        self.FN = 0
        self.FP = 0
        self.TP = 0

        #Starting testing
        for data in self.test:
            x = data[:8] #Input x of some dimension
            y = data[8] #Expected output

            #Calculate the three likelihood: universal, with respect to y=1, and with respect to y=2
            l1, l2, l3 = (
                self.getLikelihood(x, self.mean, self.var),
                self.getLikelihood(x, self.meanPY, self.varPY),
                self.getLikelihood(x, self.meanPNotY, self.varPNotY),
            )

            #Classify the output.
            predict = self.classify(l1, l2, l3, self.pY)

            #Record the accuracy about predicted output vs expected output
            if predict == 0 and y == 0:
                self.TN = self.TN + 1

            if predict == 1 and y == 0:
                self.FP = self.FP + 1

            if predict == 0 and y == 1:
                self.FN = self.FN + 1

            if predict == 1 and y == 1:
                self.TP = self.TP + 1

    '''
    Load data from files
    '''
    def loadData(self):
        testSet = np.genfromtxt("./data/test.csv", delimiter=",")
        trainSet = np.genfromtxt("./data/train.csv", delimiter=",")
        return trainSet, testSet

    '''
    Generate means
    '''
    def getMean(self, input):
        shape = np.shape(input)
        result = np.ones(shape[len(shape) - 1])
        block = shape[len(shape) - 1]
        for i in range(block):
            result[i] = np.mean(input.flatten()[i::block])
        return result

    '''
    Generate variances
    '''
    def getVar(self, input):
        shape = np.shape(input)
        result = np.ones(shape[len(shape) - 1])
        block = shape[len(shape) - 1]
        for i in range(block):
            result[i] = np.var(input.flatten()[i::block])
        return result

    '''
    Calculate likelihood using multivariate gaussian distribution 
    '''
    def getLikelihood(self, x, mean, variance):

        #Find Σ
        sigma = np.identity(x.size) * variance

        #Calculate the coefficient: 
        coefficient = 1 / (
            ((2 * np.pi) ** (x.size / 2)) * ((np.linalg.det(sigma)) ** (1 / 2))
        )

        #Calculate the exponent term
        d = sigma ** (-1)
        d[np.isinf(d)] = 0
        a = np.atleast_2d(x - mean)  # 1 x n matrix
        b = d  # n x n matrix
        c = np.transpose(np.atleast_2d(x - mean))  # n x 1 matrix

        #Return the final results
        return (coefficient * np.exp((-1 / 2) * np.linalg.multi_dot([a, b, c])))[0][0]

    '''
    Classify the result
    '''
    def classify(self, likelihood1, likelihood2, likelihood3, pY):
        return 1 if likelihood2 * pY >= likelihood3 * (1 - pY) else 0

    '''
    Calcualte metrics
    '''
    def calculateAccuracy(self):
        print((self.TP + self.TN) / (self.TN + self.TP + self.FN + self.FP))



ndb = NaiveBayesianClassifer()
ndb.calculateAccuracy()
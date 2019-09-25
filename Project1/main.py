import numpy as np

"""
An implementation of Gaussian Naive Bayesian Classifier
@author: Sotheanith Sok, Jesse Blacklock
@class: CECS 550
@instructor: Arjang Fahim
@date: 9/23/2019
"""


class GaussianNaiveBayesianClassifier(object):
    def __init__(self, shuffle=False, ratio=2.0 / 3.0):
        """Initialize the object, load data, and start to predict the result.

        Keyword Arguments:
            shuffle {bool} -- Determine if training set and test set should be randomly pick from data set. Load predetermined training set and test set by default. (default: {False})
            ratio {float} -- Determine the ratio between training set and test set. Only applicable if shuffle is true  (default: {2.0/3.0})
        """
        super().__init__()

        # Load data
        self.trainingSet, self.testSet = self.loadData(shuffle)

        # Calculate Prior probability
        self.pY = (
            np.count_nonzero(self.trainingSet[:, 8]) / np.shape(self.trainingSet)[0]
        )

        # Calculate likelihood of Y=1
        self.meanPY, self.variancePY = self.calculateMeanAndVariance(
            self.trainingSet[self.trainingSet[:, 8] == 1][:, :8]
        )

        # Calculate likelihood of Y=0
        self.meanPNotY, self.VariancePNotY = self.calculateMeanAndVariance(
            self.trainingSet[self.trainingSet[:, 8] == 0][:, :8]
        )

        # Initialize performance trackers to zero
        self.TN, self.FP, self.FN, self.TP = 0, 0, 0, 0

        # Start predicting on test set
        self.predict()

        # Calculate metrics and print them out
        self.accuracy = 0
        self.error = 0
        self.sensitivity = 0
        self.specificity = 0

        self.calculateMetrics()

    def loadData(self, shuffle=False, ratio=2.0 / 3.0):
        """Load data into our object from files

        Keyword Arguments:
            shuffle {bool} -- Determine if training set and test set should be randomly pick from data set. Load predetermined training set and test set by default. (default: {False})
            ratio {float} -- Determine the ratio between training set and test set. Only applicable if shuffle is true  (default: {2.0/3.0})
        """
        # For non-shuffling data, use predetermine testing and training sets
        if shuffle == False:
            testSet = np.genfromtxt("./data/test.csv", delimiter=",")
            trainingSet = np.genfromtxt("./data/train.csv", delimiter=",")
            return trainingSet, testSet

        # Shuffle entire dataset and split base on ratio if required
        else:
            dataSet = np.genfromtxt(
                "./data/pima-indians-diabetes.data.csv", delimiter=","
            )
            np.random.shuffle(dataSet)
            splitTarget = int(np.shape(dataSet)[0] * ratio)
            return dataSet[:splitTarget], dataSet[splitTarget:]

    def calculateMeanAndVariance(self, input):
        """Calculate means and variances column wise

        Arguments:
            input {2d array} -- An array of features set
        """
        mean = np.mean(input, axis=0)
        variance = np.var(input, axis=0)
        return mean, variance

    def calculateGaussianPDF(self, x, mean, variance):
        """Calcualte gaussian probability density of a single feature

        Arguments:
            x {float} -- Value of a feature
            mean {float} -- mean of other similar features
            variance {float} -- variance of other similar features

        Returns:
            float -- Gaussian probability density
        """
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * variance)))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent

    def predict(self):
        """
        Predicts the test set and records results
        """
        for X in self.testSet:
            probYX, probNotYX = 0, 0
            for i in range(X.size - 1):

                # Calculate P(X|Y=1)
                probXY = self.calculateGaussianPDF(
                    X[i], self.meanPY[i], self.variancePY[i]
                )
                if probXY > 0:
                    probYX = np.log(probXY) + probYX  # logarithmic estimation

                # Calculate P(X|Y=0)
                probXNotY = self.calculateGaussianPDF(
                    X[i], self.meanPNotY[i], self.VariancePNotY[i]
                )
                if probXNotY > 0:
                    # logarithmic estimation
                    probNotYX = np.log(probXNotY) + probNotYX

            # Calculate P(Y=1|X) = P(Y=1|X) * P(Y=1)
            # P(Y=1|X) = log(P(Y=1|X) * P(Y=1)) = log(P(Y=1|X)) + log(P(Y=1))
            probYX = probYX + np.log(self.pY)

            # Calculate P(Y=0|X) = P(Y=0|X) * P(Y=0)
            probNotYX = probNotYX + np.log(1 - self.pY)

            # Make the decision
            predict = 1 if probYX > probNotYX else 0

            # Record result
            if predict == 0 and X[8] == 0:
                self.TN = self.TN + 1
            elif predict == 0 and X[8] == 1:
                self.FN = self.FN + 1
            elif predict == 1 and X[8] == 0:
                self.FP = self.FP + 1
            else:
                self.TP = self.TP + 1

    def calculateMetrics(self):
        """
        Calcualte metrics such as accuracy and print out those matrics 
        """
        self.accuracy = (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN)
        self.error = (self.FP + self.FN) / (self.TP + self.FP + self.TN + self.FN)
        self.sensitivity = self.TP / (self.FN + self.TP)
        self.specificity = self.TN / (self.TN + self.FP)

        # Print results
        print("Accuracy:", self.accuracy)
        print("Error:", self.error)
        print("Sensitivity:", self.sensitivity)
        print("Specificity:", self.specificity)


# Run n times and average the results
accuracy = 0
error = 0
sensitivity = 0
specificity = 0

for i in range(100):

    # Keep track of trials
    print("Trial ", i, " :")

    # Shuffle input data and split at random ratio
    gnb = GaussianNaiveBayesianClassifier(True, np.random.random_sample())

    # Record results
    accuracy = accuracy + gnb.accuracy
    sensitivity = sensitivity + gnb.sensitivity
    error = error + gnb.error
    specificity = specificity + gnb.specificity
    print("-------------------------------------")

i=i+1
print("Final results: ")
print("Accuracy:", accuracy / i)
print("Error:", error / i)
print("Sensitivity:", sensitivity / i)
print("Specificity:", specificity / i)


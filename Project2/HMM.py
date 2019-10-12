import numpy as np
"""
An implementation of Hidden Morkov Model for evaluation problem and decoding problem
@author: Sotheanith Sok, Jesse Blacklock
@class: CECS 550
@instructor: Arjang Fahim
@date: 10/11/2019
"""

class HMM (object):
    """Hidden Morkov Model
    """

    def __init__(self, input):
        """Constructor

        Arguments:
            input {string} -- a vector of observable states
        """
        super().__init__()

        # Convert input strings to integer values
        input[input == 'yes'] = 1
        input[input == 'no'] = 2
        self.input = input.astype(np.int)

        # Performe operations
        self.loadData()
        self.createTransitionProbabilities()
        self.createEmissionProbabilities()
        self.createAlphas()
        self.runViterbi()
        self.printResult()

    def loadData(self):
        """Load data from file
        """
        # Load from files
        self.data = np.genfromtxt(
            './prompts/Project2Data.txt', delimiter=',', dtype='unicode')

        # Convert string values to integer values
        self.data[self.data == 'sunny'] = 1
        self.data[self.data == 'rainy'] = 2
        self.data[self.data == 'foggy'] = 3
        self.data[self.data == 'yes'] = 1
        self.data[self.data == 'no'] = 2
        self.data = self.data.astype(np.int)

    def createTransitionProbabilities(self):
        """Create transition probabilities from data
        """
        # Count how many unique hidden states in data
        hiddenStates = np.unique(self.data[:, 0]).size

        # Create transition matrix including hidden state 0
        self.a = np.zeros((hiddenStates+1, hiddenStates+1))

        # Counting hidden states transition
        initial = self.data[0, 0]
        target = -1
        for x in self.data[1:, 0]:
            target = x
            self.a[initial, target] = self.a[initial, target]+1
            initial = x

        # Average transitions by the number of transitions coming out of a hidden states
        self.a = np.transpose(np.transpose(self.a) / np.sum(self.a, axis=1))

        # Replace nan with 0
        self.a = np.nan_to_num(self.a)

        # Ensure that transition from hidden state 0 to hidden state 0 is one
        self.a[0, 0] = 1

    def createEmissionProbabilities(self):
        """Create emission matrix from data
        """
        # Get the number of hidden states
        hiddenStates = np.unique(self.data[:, 0]).size

        # Get the number of observable states
        observableStates = np.unique(self.data[:, 1]).size

        # Initialize emmision matrix with hidden states 0 and emission states 0
        self.b = np.zeros((hiddenStates+1, observableStates+1))

        # Start couting
        for x in self.data:
            self.b[x[0], x[1]] = self.b[x[0], x[1]]+1

        # Average emissions by the number of transitions coming out of a hidden states
        self.b = np.transpose(np.transpose(self.b) / np.sum(self.b, axis=1))

        # Replace nan with 0
        self.b = np.nan_to_num(self.b)

        # Ensure that transition from hidden state 0 to observable state 0 is one
        self.b[0, 0] = 1

    def createAlphas(self):
        """Generate alphas using forward algorithm
        """
        # Get the length of time and number of hidden states
        time = self.input.size
        hiddenStates = np.unique(self.data[:, 0]).size

        # Intialize the alpha matrix
        self.alpha = np.zeros((hiddenStates+1, time+1))

        # Populate the time 0 by the intial states (Assume starting at "sunny")
        self.alpha[:, 0] = np.array([0, 1, 0, 0])

        # Start the calculation
        for t in range(1, time+1):
            for w in range(hiddenStates+1):
                s = np.sum(self.alpha[:, t-1] * self.a[:, w]
                           ) * self.b[w, self.input[t-1]]
                self.alpha[w, t] = s

    def runViterbi(self):
        """Generate scores using Viterbi algorithm  
        """
        # Get the length of time and the number of hidden states
        time = self.input.size
        hiddenStates = np.unique(self.data[:, 0]).size

        # Initialize the scores matrix
        self.score = np.zeros((hiddenStates+1, time+1))

        # Populate scores at time 0
        self.score[:, 0] = np.array([0, 1, 0, 0])

        # Begin the Viterbi algorithm
        for t in range(1, time+1):
            for w in range(hiddenStates+1):
                s = np.max(self.score[:, t-1] * self.a[:, w]
                           ) * self.b[w, self.input[t-1]]
                self.score[w, t] = s

    def printResult(self):
        """Print results
        """
        # Transition probabilities
        print("Transition Probabilities: ")
        print(self.a)

        # Emission probabilities
        print("Emission Probabilities: ")
        print(self.b)

        # Overall probabilites of generating the sequence of observable states
        inp = self.input.astype(np.unicode)
        inp[inp == '1'] = "yes"
        inp[inp == '2'] = "no"
        print("For input: ", inp)
        print("Probability of such evidents generated from this HMM: ",
              np.sum(self.alpha[:, self.input.size]))

        # Most likely sequence of hidden states to generate the given observable states
        seq = np.argmax(self.score, axis=0)
        seq = seq.astype(np.unicode)
        seq[seq == "1"] = "sunny"
        seq[seq == "2"] = "rainy"
        seq[seq == "3"] = "foggy"
        print(
            "Most likely sequence of hidden states to generate the above evidents: \n", seq)


HMM(np.array(['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']))

from math import exp
from pynq.mllib_accel.accelerators import LR_Accel
from time import time

class LogisticRegression(object):
    """
    Multiclass Logistic Regression
    
    One versus rest: The algorithm compares every class with all the remaining classes, 
                     building a model for every class.
    """

    def __init__(self, numClasses, numFeatures, weights = None):
        """
            :param numClasses:     Number of possible outcomes.

            :param numFeatures:    Dimension of the features.

            :param weights:        Weights computed for every feature.
        """

        self.numClasses = numClasses
        self.numFeatures = numFeatures
        self.weights = weights

    def train(self, trainFile, chunkSize, alpha = 0.25, iterations = 5, _accel_ = 0):
        """ 
        Train a logistic regression model on the given data.
   
            :param trainFile:      The training data file.
            
            :param chunkSize:     Size of each data chunk. 

            :param alpha:         The learning rate (default: 0.25).

            :param iterations:    The number of iterations (default: 5).
            
            :param _accel_:       0: SW-only, 1: HW accelerated (deafult: 0).
        """

        def process_trainFile(lines):
            data = []
            tmp = []

            for i, line in enumerate(lines):
                line = line.split(",")

                for k in range(0, self.numClasses):
                    if k == int(line[0]):
                        tmp.append(1.0)
                    else:
                        tmp.append(0.0)
                tmp.append(1.0)
                for j in range(0, self.numFeatures):
                    tmp.append(float(line[j + 1]))

                if (i + 1) % chunkSize == 0:
                    data.append(tmp)
                    tmp = []

            return data

        def gradients_kernel(data, weights):
            chunkSize = int(len(data) / (self.numClasses + (1 + self.numFeatures)))

            gradients = []
            for k in range (0, self.numClasses):
                for j in range (0, (1 + self.numFeatures)):
                    gradients.append(0.0)

            for i in range (0, chunkSize):
                labels = []
                for k in range (0, self.numClasses):
                    labels.append(float(data[i * (self.numClasses + (1 + self.numFeatures)) + k]))
                features = []
                for j in range (0, (1 + self.numFeatures)):
                    features.append(float(data[i * (self.numClasses + (1 + self.numFeatures)) + self.numClasses + j]))

                for k in range (0, self.numClasses):
                    dot = 0.0
                    for j in range (0, (1 + self.numFeatures)):
                        dot += float(weights[k * (1 + self.numFeatures) + j]) * features[j]

                    dif = 1.0 / (1.0 + exp(-dot)) - labels[k]

                    for j in range (0, (1 + self.numFeatures)):
                        gradients[k * (1 + self.numFeatures) + j] += dif * features[j]

            return gradients

        print("    * LogisticRegression Training *")

        trainFile = process_trainFile(trainFile)         
        numSamples = len(trainFile) * chunkSize 

        print("     # numSamples:               " + str(numSamples))
        print("     # chunkSize:                " + str(chunkSize))
        print("     # numClasses:               " + str(self.numClasses))
        print("     # numFeatures:              " + str(self.numFeatures))
        print("     # alpha:                    " + str(alpha))
        print("     # iterations:               " + str(iterations))

        start = time()

        # Batch Gradient Descent Algorithm
        if self.weights is None:
            self.weights = []
            for k in range(0, self.numClasses):
                for j in range(0, (1 + self.numFeatures)):
                    self.weights.append(0.0)
        
        if _accel_:
            accel = LR_Accel(chunkSize, self.numClasses, self.numFeatures)    
        
        for t in range(0, iterations):
            gradients = [0] * (self.numClasses * (self.numFeatures + 1))
            for c in range(0, len(trainFile)):
                if _accel_:
                    gradients = [a + b for a, b in zip(gradients, accel.gradients_kernel(trainFile[c], self.weights))]
                else:
                    gradients = [a + b for a, b in zip(gradients, gradients_kernel(trainFile[c], self.weights))]

            for k in range(0, self.numClasses):
                for j in range(0, (1 + self.numFeatures)):
                    self.weights[k * (1 + self.numFeatures) + j] -= (alpha / numSamples) * gradients[k * (1 + self.numFeatures) + j]

        if _accel_:
            accel.__del__()
                    
        end = time()
        if _accel_:
            print("! Time running training in hardware: " + str(round(end - start, 3)) + " sec")
        else:
            print("! Time running training in software: " + str(round(end - start, 3)) + " sec")

        return self.weights

    def test(self, testFile):
        """
        Test a logistic regression model on the given data.

            :param testFile:    The testing data file.
        """

        def process_testFile(lines):
            data = []

            for line in lines:
                line = line.split(",")

                tmp = []
                tmp.append(int(line[0]))
                tmp.append(1.0)
                for j in range(0, self.numFeatures):
                    tmp.append(float(line[j + 1]))

                data.append(tmp)

            return data

        print("    * LogisticRegression Testing *")

        testFile = process_testFile(testFile) 

        true = 0
        for c in range(0, len(testFile)):
            true += 1 if testFile[c][0] == self.predict(testFile[c][1:]) else 0
        false = len(testFile) - true

        print("     # accuracy:                 " + str(float(true) / float(true + false)) + "(" + str(true) + "/" + str(true + false) + ")")
        print("     # true:                     " + str(true))
        print("     # false:                    " + str(false))

    def predict(self, features):
        """
        Predict values for a single data point using the model trained.

            :param features:    Features to be labeled.
        """

        max_probability = -1.0
        prediction = -1

        for k in range (0, self.numClasses):
            dot = 0.0
            for j in range (0, (1 + self.numFeatures)):
                dot += float(features[j]) * float(self.weights[k * (1 + self.numFeatures) + j])

            probability = 1.0 / (1.0 + exp(-dot))

            if probability > max_probability:
                max_probability = probability
                prediction = k

        return prediction

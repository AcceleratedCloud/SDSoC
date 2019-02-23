import numpy as np
from math import exp ,sqrt ,pi, log , inf
from pyspark import RDD
from time import time
from itertools import tee
from functools import reduce
from .accelerators.Naivebayes import cma_train, cma_predict, predictionNB_kernel_accel, trainingNB_kernel_accel, cmf_train, cmf_predict

__all__ = ['Naivebayes']

class Naivebayes(object):
    """
    Multiclass Naive Bayes Model.

        :param numClasses:     Number of possible outcomes.

        :param numFeatures:    Dimension of the features.

    """


    def __init__(self, numClasses, numFeatures, trainPack = None ):
        self.numClasses = numClasses
        self.numFeatures = numFeatures
        self.trainPack = trainPack

    def train(self, trainRDD, _accel_ = 0):  
        """ 
        Train a naive bayes model on the given data. 
   
            :param trainRDD:        The training data, a list of LabeledPoint.
  
            :param _accel_:          0: SW-only, 1: HW accelerated (deafult: 0).

            :note:                   Labels used in naive bayes should be 
                                     {0, 1, ..., k - 1} for k classes classification problem.
        """        

        def array_red(a,b):
            adding = [tuple([tuple(row) if not isinstance(row,np.float32) else row for row in np.array(aa)+np.array(bb)]) for aa, bb in zip(a, b)]
            return adding

        def sorting(trainRDD):
            trainRDD = sorted(trainRDD , key = lambda x: x.label)
            trainRDDcount = list(map(lambda x: x.label, trainRDD))

            offset=[]
            for i in range(self.numClasses):
                offset.append(trainRDDcount.count(float(i))) 
                  
            return trainRDD, offset

        def trainingNB_kernel(data, offset):
            offset_counter = 0

            priors = np.zeros((self.numClasses))
            variances = np.zeros((self.numClasses,self.numFeatures))
            means = np.zeros((self.numClasses,self.numFeatures))
            sums_x = np.zeros((self.numClasses,self.numFeatures))
            sq_sums_x = np.zeros((self.numClasses,self.numFeatures))
            for line in data: 
                label = int(line.label)
                if offset_counter < offset[label]-1:
                    for i in range(0, self.numFeatures):
                        sums_x[label][i] += line.features[i]
                        sq_sums_x[label][i] += line.features[i]*line.features[i]
                    offset_counter += 1                 
                else:
                    priors[label] = offset[label]/sum(offset)
                    for i in range(0, self.numFeatures):
                        means[label][i] = sums_x[label][i] / offset[label]
                        variances[label][i] = (sq_sums_x[label][i] / offset[label]) - (means[label][i]*means[label][i])
                    offset_counter = 0  

            trainPack = [means, variances, priors]

            return trainPack

        # Reduction of multiclass classification to binary classification.
        # Performs reduction using one against all strategy (OneVsRest).
        # For a multiclass classification with k classes, train k models (one per class).
        print("* NaiveBayes Training *")
        numBuffers = 1
        if (_accel_):
            trainRDD = list(cma_train(trainRDD))
            numBuffers = len(trainRDD)
        else:
            # Sort Data by type of class and counting overall features per class
            trainRDD, offset = sorting(trainRDD)


        print("     # numBuffers:               {:d}".format(numBuffers))
        print("     # numClasses:               {:d}".format(self.numClasses))
        print("     # numFeatures:              {:d}".format(self.numFeatures))
        print("     # Accelerated:              {0}".format(bool(_accel_)))


        #start = time()   

        if (_accel_):
            trainPackage = list(map(lambda data: trainingNB_kernel_accel(data), trainRDD))
            trainPackage = reduce(lambda a, b: array_red(a, b),trainPackage)
            trainPackage = [tuple([tuple(row/numBuffers) if not isinstance(row,np.float32) else row/numBuffers for row in np.array(aa)]) for aa in trainPackage]
            self.trainPack = list(trainPackage)
        else:
            trainPackage = trainingNB_kernel(trainRDD, offset)
            self.trainPack = list(trainPackage)
        #end = time() 
        

        if (_accel_):
            map(cmf_train,trainRDD)      

    def test(self, testRDD, _accel_ = 0):
            """
            Test a naive bayes model on the given data.

                :param testRDD:    The testing data, a list of LabeledPoint.

                :note:             Labels used in Naive Bayes should be 
                                   {0, 1, ..., k - 1} for k classes classification problem.
            """

            # Each example is scored against all k models and the model with highest probability
            # is picked to label the example.

            print("* NaiveBayes Testing *")


            if (_accel_):
                address = cma_predict(self.trainPack)
            else:
                testRDD = testRDD

            #start = time()       

            if (_accel_): 
                true = map(lambda data: 1 if data.label == predictionNB_kernel_accel(address, data.features) else 0,testRDD)
            else:
                true = list(map(lambda data: 1 if data.label == self.predict(data.features) else 0, testRDD))
            true = reduce(lambda a, b: a + b,true)
            false = len(testRDD) - true

            #end = time() 

            print("     # accuracy:                 {:.3f} ({:d}/{:d})".format(true / (true + false), true, true + false))
            print("     # true:                     {:d}".format(true))
            print("     # false:                    {:d}".format(false))

            if (_accel_):
                cmf_train(address) 

            return self

    def predict(self, data):
        """
        Predict values for a single data point using the model trained.

            :param features:    Features to be labeled.
        """
        prediction = 0     
        max_likelihood = -inf
        for label in range(0, self.numClasses):
            numerator = log(self.trainPack[2][label])
            for j in range(0 , self.numFeatures):
                if  self.trainPack[1][label][j] < 0.00005: 
                    numerator += 0.0                      
                else:
                    numerator += log(1 / sqrt(2 * pi * self.trainPack[1][label][j])) + ((-1*(data[j] - self.trainPack[0][label][j])**2) / (2 * self.trainPack[1][label][j]))             
            if numerator > max_likelihood :
                max_likelihood = numerator
                prediction = label

        return prediction
        
    def save(self, path, stats):
        """
        Save this model to the given path.
        """

        np.savetxt(path, self.trainPack[stats],newline='n')


    def load(self, path):
        """
        Load a model from the given path.
        """

        self.trainPack = np.loadtxt(path)

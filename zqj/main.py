import random
import numpy as np
import time
from NeuralNetwork.data_utils import *
from NeuralNetwork.admm import *


def getStatistics(Xtr, Ytr, Xte, Yte, weightConsWeight, activConsWeight, iterNum, epsilon):
    # Initialize a 3-layer neural network with specified neuron dimension, 
    # the first dim is determined by the size of input dataset
    neuronDim = [0, 100, 150]
    network = NeuralNetwork(3, neuronDim)
    classNum = 10
    
    # Initialize weight matrix W of first layer 
    W = np.random.randn(Xtr.shape[1], classNum) * epsilon

    # Train
    tic = time.time()
    weight = network.train(W, Xtr, Ytr, weightConsWeight, activConsWeight, iterNum, epsilon)
    toc = time.time()
    print 'Total training time: %fs' % (toc - tic)
    
    # Predict with the trained feature weight
    Ypred = network.predict(weight, Xte)
    print 'Prediction accuracy: %f' %np.mean(Ypred == Yte)


def miniPatchTest(trNum, teNum):
    # Subsample the data for more efficient code execution 
    trainNum = trNum
    mask = range(trainNum)
    Xtr = X_train[mask]
    Ytr = Y_train[mask]
    
    testNum = teNum
    mask = range(testNum)
    Xte = X_test[mask]
    Yte = Y_test[mask]
    
    # Reshape the image data into rows
    Xtr = np.reshape(Xtr, (Xtr.shape[0], -1))
    Xte = np.reshape(Xte, (Xte.shape[0], -1))
    
    # Specify weight coefficients of two regularization term, iteration of ADMM updates
    weightConsWeight = 1
    activConsWeight = 10
    iterNum = 20
    epsilon = 0.01
    getStatistics(Xtr, Ytr, Xtr, Ytr, weightConsWeight, activConsWeight, iterNum, epsilon)
 

# Main thread
# Load Mnist Data
mnistDir = "NeuralNetwork/MnistData"
X_train,Y_train,X_test,Y_test = getMnistData(mnistDir)

train = np.array([10]) # Trainning amount array
for i in train:
    print "Training data: ", i
    miniPatchTest(i,10)


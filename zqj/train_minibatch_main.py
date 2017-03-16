import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s' % time.strftime("%x")

mnistDir = "NeuralNetwork/MnistData"
(batchSize, testSize, valSize)=(100, 100, 0)
datasets = getMnistDataSets(mnistDir,valSize=valSize)
train = datasets['train']
test = datasets['train']
if valSize != 0:
    validation = datasets['validation']
else: 
    validation = None
Xte, Yte = test.nextBatch(testSize)

# Initialize Parameters
hiddenLayer = [300]
classNum = 10 
epsilon= 0.0001 
network = NeuralNetwork(train, validation, classNum, hiddenLayer, epsilon, batchSize=batchSize, valSize=valSize)

weightConsWeight = 1
activConsWeight = 10
growingStep = 1
iterOutNum =1
iterInNum = 10
hasLambda = True
calLoss = False
traditional = True

# Train 
tic = time.time()
network.trainWithMiniBatch(weightConsWeight, activConsWeight, growingStep, 
                           iterOutNum, iterInNum, hasLambda, calLoss=calLoss, 
                           lossType='smx', minMethod='prox', tau=1, ite=10, 
                           regWeight=1.0, dampWeight=0.0, evaluate=True,traditional = traditional)

toc = time.time()
print 'Total training time: %fs' % (toc - tic)

# Predictf
Ypred,z = network.predict(Xte)
print ',,,',Ypred.shape
print 'Prediction accuracy: %f' %np.mean(Ypred == Yte)

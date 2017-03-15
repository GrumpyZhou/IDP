import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s with minibatch' % time.strftime("%x")

(batchSize, testSize, valSize)=(300, 10000, 0)
mnistDir = "NeuralNetwork/MnistData"
datasets = getMnistDataSets(mnistDir,valSize=valSize)

#mnistDir = "NeuralNetwork/benchmarkData/mnistDataset.mat"
#datasets = getDataSetsFromMat(mnistDir, valSize=valSize)
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
epsilon= 0.01 
network = NeuralNetwork(train, validation, classNum, hiddenLayer, epsilon, batchSize=batchSize, valSize=valSize)

weightConsWeight = 0.001
activConsWeight = 0.001
growingStep = 10
iterOutNum = 500
iterInNum = 3
regWeight = 1.0
prox_ite = 10
traditional = True
hasLambda = True
calLoss = False

#Logging 
print 'Config: lambda:%s epsilon:%f in_iter:%d out_iter:%d batchsz:%d prox_it:%d'
      %(hasLambda,epsilon,iterInNum, iterOutNum, batchSize, prox_ite)
print 'weightConsWeight:%f activConsWeight:%f growingStep:%f regweight:%f traditional:%s'
      %(weightConsWeight,activConsWeight,growingStep, regWeight, traditional)

# Train 
tic = time.time()
network.trainWithMiniBatch(weightConsWeight, activConsWeight, growingStep, 
                           iterOutNum, iterInNum, hasLambda, calLoss=calLoss, 
                           lossType='smx', minMethod='prox', tau=1, ite=prox_ite, 
                           regWeight=regWeight, dampWeight=0.0, evaluate=True, traditional = traditional)
toc = time.time()
print 'Total training time: %fs' % (toc - tic)

# Predictf
Ypred,z = network.predict(Xte)
print 'Prediction accuracy: %f' %np.mean(Ypred == Yte)

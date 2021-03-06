import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s with minibatch' % time.strftime("%x")

(batchSize, testSize, valSize)=(400, 10000, 0)
#mnistDir = "NeuralNetwork/MnistData"
#datasets = getMnistDataSets(mnistDir,valSize=valSize)

mnistDir = "NeuralNetwork/benchmarkData/mnistDataset.mat"
#cifarDir = "NeuralNetwork/benchmarkData/cifarDataset.mat"
#spiralEasyDir = "NeuralNetwork/benchmarkData/spiralEasyDataset.mat" 

datasets = getDataSetsFromMat(mnistDir, valSize=valSize)
train = datasets['train']
test = datasets['test']
if valSize != 0:
    validation = datasets['validation']
else: 
    validation = None
Xte, Yte = test.nextBatch(testSize)
Xtr, Ytr = train.images, train.labels


# Initialize Parameters
hiddenLayer = [300]
classNum = 10
epsilon= 0.1 #dev 
network = NeuralNetwork(train, validation, classNum, hiddenLayer, epsilon, batchSize=batchSize, valSize=valSize)

weightConsWeight = 0.000001
activConsWeight = 0.000001
growingStep = 1.05
iterOutNum = 400
iterInNum = 4
hasLambda = True
calLoss = False
regWeight = 0.001
prox_ite = 10


#Logging 

print 'Config: lambda:%s epsilon:%f in_iter:%d out_iter:%d batchsz:%d prox_it:%d'%(hasLambda,epsilon,iterInNum, iterOutNum, batchSize, prox_ite)
print 'weightConsWeight:%.10f activConsWeight:%.10f growingStep:%f regweight:%f'%(weightConsWeight,activConsWeight,growingStep, regWeight)

# Train 
tic = time.time()
network.trainWithMiniBatch(weightConsWeight, activConsWeight, growingStep, 
                           iterOutNum, iterInNum, hasLambda, calLoss=calLoss, 
                           lossType='smx', minMethod='prox', tau=1, ite=prox_ite, 
                           regWeight=regWeight, dampWeight=0.0, evaluate=True)

toc = time.time()
print 'Total training time: %fs' % (toc - tic)

# Predict test
Yte_pred = network.predict(Xte)
print 'Prediction of test accuracy: %f' %np.mean(Yte_pred == Yte)

# Predict train
Ytr_pred = network.predict(Xtr)
print 'Prediction of train accuracy: %f' %np.mean(Ytr_pred == Ytr)

"""
# For visualization
L = len(hiddenLayer)
if calLoss:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(network.dataLoss)
    fig.savefig('fig/admm.png')
"""

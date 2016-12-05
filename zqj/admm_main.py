import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

def getMiniPatch(X_train, Y_train, X_test, Y_test, trNum, teNum, transposed):
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
    
    return Xtr.T, Xte.T, Ytr, Yte




print '\n\nTesting date:  %s' % time.strftime("%x")

# Load Mnist Data
mnistDir = "NeuralNetwork/MnistData"
X_train,Y_train,X_test,Y_test = getMnistData(mnistDir)

(trNum,teNum) = (6000,1000)
i = 0
X_tr, X_te, Y_tr, Y_te = getMiniPatch(X_train, Y_train, X_test, Y_test, trNum, teNum, 1)
print 'Xtr: ', X_tr.shape, 'Xte: ', X_te.shape, 'Ytr: ', Y_tr.shape, 'Yte: ', Y_te.shape

# Initialize networkfrom datetime import datetime, date, time
hiddenLayer = [300,150]
classNum = 10 
epsilon= 0.0001 
network = NeuralNetwork(X_tr, Y_tr, classNum, hiddenLayer, epsilon)

# Train param
weightConsWeight = 10
activConsWeight = 15
iterNum = 120
hasLambda = False
calLoss = False

print 'Config: epsilon:%f iter:%d '%(epsilon,iterNum)
""" 
Input:
weightConsWeight, activConsWeight
iterNum:    iteration to perform Admm updates
hasLambda:  whether include Lambda update
lossType:   one of {'hinge', 'msq', 'smx'}, default is 'smx'
minMethod:  if lossType is 'smx', the method to minimize the zLastUpdate has to be specified (for it's not in closed form), 
            it can be one of {'prox','gd','newton'}, default is 'prox';
tau, ite:   if lossType is 'smx', the step size and iteration of gradient descent/proximal gradient have to be specified, 
            default: tau=0.01, ite=25; 
"""
tic = time.time()
network.train2(weightConsWeight, activConsWeight, iterNum, hasLambda, 
                              calLoss, lossType = 'smx', minMethod = 'prox', tau= 0.01, ite= 25)
toc = time.time()
print 'Total training time: %fs' % (toc - tic)
# Predict
Ypred,z = network.predict(X_te)
print 'Prediction accuracy: %f' %np.mean(Ypred == Y_te)

dataLoss = network.getFinalDataLoss(beta=weightConsWeight, gamma=activConsWeight,lossType='smx')
print 'Final trained data loss: %f' % dataLoss

#print 'Saving weight...'
#network.saveWeight()

'''
network2 = NeuralNetwork(X_tr, Y_tr, classNum, hiddenLayer, epsilon)
tic = time.time()
network.trainWithoutMiniBatch(weightConsWeight, activConsWeight, iterNum, hasLambda, 
                              calLoss, lossType = 'smx', minMethod = 'prox', tau= 0.01, ite= 25)
toc = time.time()
print 'Total training time: %fs' % (toc - tic)
# Predict
Ypred,z = network.predict(X_te)
print 'Prediction accuracy: %f' %np.mean(Ypred == Y_te)

dataLoss = network.getFinalDataLoss(beta=weightConsWeight, gamma=activConsWeight,lossType='smx')
print 'Final trained data loss: %f' % dataLoss
'''

# For visualization
L = len(hiddenLayer)
if calLoss:
    fig = plt.figure()
    gs = gridspec.GridSpec(4,L)

    dataLoss = fig.add_subplot(gs[0,:])
    dataLoss.set_title('loss function')
    dataLoss.plot(network.dataLoss, 'k-')

    for l in range(0,L):
        aloss = fig.add_subplot(gs[1,l])
        aloss.set_title('constraint a = hz, layer = %d' % (l+1))
        aloss.plot(network.aConstrLoss[l], 'g-')

        zloss = fig.add_subplot(gs[2,l])
        zloss.set_title('constraint z = wa, layer = %d' % (l+1))
        zloss.plot(network.zConstrLoss[l], 'b-')

    zLLoss = fig.add_subplot(gs[3,:])
    zLLoss.set_title('output layer zL')
    zLLoss.plot(network.zConstrLoss[L], 'k-')

    plt.tight_layout()
    plt.show()
    




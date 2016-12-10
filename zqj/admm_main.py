import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s' % time.strftime("%x")

# Load Mnist Data
(trNum,teNum) = (100,100)
mnistDir = "NeuralNetwork/MnistData"
datasets = getMnistDataSets(mnistDir,valSize=0)
train = datasets['train']
test = datasets['train']

X_tr, Y_tr = train.images[:,range(trNum)], train.labels[range(trNum)]
X_te, Y_te = test.images[:,range(teNum)], test.labels[range(teNum)]
print 'Xtr: ', X_tr.shape, 'Xte: ', X_te.shape, 'Ytr: ', Y_tr.shape, 'Yte: ', Y_te.shape

i = 0

# Initialize networkfrom datetime import datetime, date, time
hiddenLayer = [300]
classNum = 10 
epsilon= 0.00001 
network = NeuralNetwork(X_tr, Y_tr, classNum, hiddenLayer, epsilon)

# Train param
weightConsWeight = 10
activConsWeight = 15
iterNum = 50
hasLambda = False
calLoss = False

print 'Config: lambda:%s epsilon:%f iter:%d '%(hasLambda,epsilon,iterNum)
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
    




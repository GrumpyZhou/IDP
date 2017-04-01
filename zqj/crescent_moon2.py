import random
import numpy as np
import time
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s without minibatch' % time.strftime("%x")

# Load Mnist Data
(trSize, teSize, valSize) = (6867, 3133, 0)
moonDir = "NeuralNetwork/benchmarkData/crescentMoonDataset.mat"
datasets = getDataSetsFromMat(moonDir, valSize=valSize)

train = datasets['train']
test = datasets['test']
if valSize != 0:
    validation = datasets['validation']
else: 
    validation = None

X_tr, Y_tr = train.images[:,range(trSize)], train.labels[range(trSize)]
X_te, Y_te = test.images[:,range(teSize)], test.labels[range(teSize)]
print 'Xtr: ', X_tr.shape, 'Xte: ', X_te.shape, 'Ytr: ', Y_tr.shape, 'Yte: ', Y_te.shape

# Initialize networkfrom datetime import datetime, date, time
hiddenLayer = [10, 10, 10, 10, 10]
classNum = 2
epsilon= 0.01 
network = NeuralNetwork(train, validation, classNum, hiddenLayer, epsilon, batchSize=trSize, valSize=valSize)

# Train param
weightConsWeight = 0.001 
activConsWeight = 0.001
growingStep = 1.02
iterNum = 450
hasLambda = True 
calLoss = False
regWeight = 0.001


print 'Config: lambda:%s epsilon:%f iter:%d'%(hasLambda,epsilon,iterNum)
print 'weightConsWeight:%.8f activConsWeight:%.8f growingStep:%f regweight:%f'%(weightConsWeight,activConsWeight,growingStep, regWeight)
tic = time.time()
network.trainWithoutMiniBatch(weightConsWeight, activConsWeight, growingStep, iterNum, hasLambda, 
                              calLoss, lossType = 'smx', minMethod = 'prox', tau= 0.01, ite= 25, 
                              regWeight=regWeight, dampWeight=0.0, evaluate=True)
toc = time.time()
print 'Total training time: %fs' % (toc - tic)
# Predict test
Y_te_pred = network.predict(X_te)
print 'Prediction of test accuracy: %f' %np.mean(Y_te_pred == Y_te)

# Predict train
Y_tr_pred = network.predict(X_tr)
print 'Prediction of train accuracy: %f' %np.mean(Y_tr_pred == Y_tr)

# Save weight as .mat
weight = {}
for i in range(1,len(network.W)):
    name = 'w%d'%i
    weight[name] = network.W[i]
print weight.keys()
sio.savemat('./crescent2.mat',weight)


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
    




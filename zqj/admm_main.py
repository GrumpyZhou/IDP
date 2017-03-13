import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s' % time.strftime("%x")

# Load Mnist Data
(trSize, teSize, valSize) = (100, 100, 0)

mnistDir = "NeuralNetwork/MnistData"
datasets = getMnistDataSets(mnistDir,valSize=valSize)
train = datasets['train']
test = datasets['train']
if valSize != 0:
    validation = datasets['validation']
else: 
    validation = None

X_tr, Y_tr = train.images[:,range(trSize)], train.labels[range(trSize)]
X_te, Y_te = test.images[:,range(teSize)], test.labels[range(teSize)]
print 'Xtr: ', X_tr.shape, 'Xte: ', X_te.shape, 'Ytr: ', Y_tr.shape, 'Yte: ', Y_te.shape

# Initialize networkfrom datetime import datetime, date, time
hiddenLayer = [300]
classNum = 10 
epsilon= 0.00001 
network = NeuralNetwork(train, validation, classNum, hiddenLayer, epsilon, batchSize=trSize, valSize=valSize)

# Train param
weightConsWeight = 0.001
activConsWeight = 0.001
growingStep = 1.02
iterNum = 50
hasLambda = True 
calLoss = False

print 'Config: lambda:%s epsilon:%f iter:%d'%(hasLambda,epsilon,iterNum)
print 'weightConsWeight:%f activConsWeight:%f growingStep:%f'%(weightConsWeight,activConsWeight,growingStep)
tic = time.time()
network.trainWithoutMiniBatch(weightConsWeight, activConsWeight, growingStep, iterNum, hasLambda, 
                              calLoss, lossType = 'smx', minMethod = 'prox', tau= 0.01, ite= 25, 
                              regWeight=1.0, dampWeight=0.0, evaluate=False)
toc = time.time()
print 'Total training time: %fs' % (toc - tic)
# Predict
Ypred,z = network.predict(X_te)
print 'Prediction accuracy: %f' %np.mean(Ypred == Y_te)

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
    




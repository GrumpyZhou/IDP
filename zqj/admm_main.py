import random
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from NeuralNetwork.data_utils import *
from NeuralNetwork.neural_network import *

print '\n\nTesting date:  %s without minibatch' % time.strftime("%x")

# Load Mnist Data
(trSize, teSize, valSize) = (6867, 3133, 0)
"""
mnistDir = "NeuralNetwork/MnistData"
datasets = getMnistDataSets(mnistDir,valSize=valSize)
"""
#mnistDir = "NeuralNetwork/benchmarkData/mnistDataset.mat"
#cifarDir = "NeuralNetwork/benchmarkData/cifarDataset.mat"
#moonDir = "NeuralNetwork/benchmarkData/crescentMoonDataset.mat"

spiralEasyDir = "NeuralNetwork/benchmarkData/spiralEasyDataset.mat" 
datasets = getDataSetsFromMat(spiralEasyDir, valSize=valSize)

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
hiddenLayer =[50,50]#,10,10,10]
classNum = 2
epsilon= 0.001 # Glorot if 0.0
network = NeuralNetwork(train, validation, classNum, hiddenLayer, epsilon, batchSize=trSize, valSize=valSize)

# Train param
weightConsWeight = 0.1 
activConsWeight = 0.1
growingStep = 1.00001
iterNum = 200
hasLambda = True 
calLoss = False
regWeight = 0.01 


print 'Config: lambda:%s epsilon:%f iter:%d'%(hasLambda,epsilon,iterNum)
print 'weightConsWeight:%.10f activConsWeight:%.10f growingStep:%f regweight:%f'%(weightConsWeight,activConsWeight,growingStep, regWeight)
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

# For visualization
"""
L = len(hiddenLayer)
if calLoss:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(network.dataLoss)
    fig1.savefig('fig/energy_no_lambda.png')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(network.aConstrLoss[0])
    fig2.savefig('fig/aConstraint_no_lambda.png')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(network.zConstrLoss[0])
    fig3.savefig('fig/zConstraint_no_lambda.png')
"""

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
    




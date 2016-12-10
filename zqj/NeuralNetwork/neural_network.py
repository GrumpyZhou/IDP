import numpy as np
from random import shuffle
import time
import datetime
from data_utils import DataSet

class NeuralNetwork():
    """
    A neural network with L layers, for each layer the dimension of neurons is specified in Dim[],
    Dim[0] is later filled by the input dimension.
    """
    def __init__(self, Xtr, Ytr, classNum, hiddenLayer, epsilon):
        """
        Input:
        - Xtr: A numpy array of shape (D, N) containing a minibatch of data
        - Ytr: A numpy array of shape (N,) containing training labels
        - classNum: The number of label classes
        - hiddenLayer: A list specifys dimension of hidden layer
        - epsilon: The coefficient for initialize random weight matrix
        """
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.classNum = classNum
        self.trainNum = Xtr.shape[1]
        self.hiddenLayer = hiddenLayer
        self.epsilon = epsilon
        
        self.W = []
        self.zL = np.zeros((0))  
        
        self.totalLoss = [] # total cost
        self.dataLoss = [] # cost from loss function
        self.aConstrLoss = [] # cost from constraint a = h(z)
        self.zConstrLoss = [] # cost from constraint z = wa
        self.lagraLoss = [] # cost from lagrange term
        for l in range(1,len(self.hiddenLayer)+ 1):
            self.aConstrLoss.append([])
            self.zConstrLoss.append([])
        self.zConstrLoss.append([]) # cost from output layer zL

        print "Initializing a neural network with : ", len(hiddenLayer)," hidden layers, hidden layer dimension:", hiddenLayer
        

    def saveWeight(self):
        w = self.W
        today = datetime.datetime.now().strftime ("%Y%m%d")
        np.savez('test/saved_weight/weight %s.npz' % today, w1=w[1], w2=w[2])

    def loadWeight(self, path=None):
        w = [0]
        with np.load("weight.npz") as data:
            w.append(data['w1'])
            w.append(data['w2'])
        self.W = w

    def initNetwork(self, Xtr, classNum, hiddenLayer, epsilon, initW):
        """ 
        Return:
        - a: Activation list for each layer [a0, a1, a2]
        - z: A z list for each layer [0, z1, z2, z3]
        - w: weight list for each layer  [0, w1, w2, w3]
        """
        L = len(hiddenLayer)
        a = [Xtr]
        z = [np.zeros((0))]
        w = [np.zeros((0))]
        
        if initW == None:
            initW = [np.zeros((0))]
            for l in range(0, L):
                initW.append(epsilon*np.random.randn(hiddenLayer[l], a[l].shape[0]))
            initW.append(epsilon*np.random.randn(classNum, hiddenLayer[L-1]))            
        
        for l in range(0, L):
            w.append(initW[l+1])
            z.append(w[l+1].dot(a[l]))
            a.append(self.ReLU(z[l+1]))

        w.append(initW[L+1])            
        z.append(w[L+1].dot(a[L]))
	return a, z, w

    def initNetworkRandom(self, trainNum, classNum, hiddenLayer, epsilon):
        """ 
        Return:
        - a: Activation list for each layer [a0, a1, a2]
        - z: A z list for each layer [0, z1, z2, z3]
        - w: weight list for each layer  [0, w1, w2, w3]
        """
        L = len(self.hiddenLayer)
        w = [np.zeros((0))]
        a = [self.Xtr]
        z = [np.zeros((0))]

        for l in range(0, L):
            w.append(epsilon*np.random.randn(hiddenLayer[l], a[l].shape[0]))
            z.append(epsilon*np.random.randn(hiddenLayer[l], trainNum))
            a.append(epsilon*np.random.randn(hiddenLayer[l], trainNum))
                
        w.append(epsilon*np.random.randn(classNum, hiddenLayer[L-1]))
        z.append(epsilon*np.random.randn(classNum, trainNum)) 
        return a, z, w

    def admmUpdate(self, y, a, z, w, L, iter, beta, gamma, hasLambda, calLoss, lossType, minMethod, tau, ite, Lambda = None):
        # One ADMM updates
        for i in range(iter):
            t1 = time.time()
            
            # Walk through 1~L-1 layer network
            for l in range(1, L):
                # w update
                #w[l] = np.linalg.lstsq(a[l-1].T,z[l].T)[0].T
                w[l] = z[l].dot(np.linalg.pinv(a[l-1]))                
               
                # a update
                wNtr = w[l+1].T
                a[l] = np.linalg.inv(beta * wNtr.dot(w[l+1]) + gamma * np.identity(wNtr.shape[0])).dot(beta * wNtr.dot(z[l+1]) + gamma * self.ReLU(z[l]))   
                #a[l] = np.linalg.lstsq(beta * wNtr.dot(w[l+1]) + gamma * np.identity(wNtr.shape[0]),np.identity(wNtr.shape[0]))[0].T.dot(beta * wNtr.dot(z[l+1]) + gamma * self.ReLU(z[l])) 
               
                # z update
                z[l] = self.zUpdate(beta, gamma, w[l].dot(a[l-1]), a[l])
                           
            # L-layer
            # w update
            #w[L] = np.linalg.lstsq(a[L-1].T,z[L].T)[0].T            
            w[L] = z[L].dot(np.linalg.pinv(a[L-1]))
            
            # zL update
            waL =  w[L].dot(a[L-1])

            #print 'Train model: lossType %s, minMethod %s', (lossType, minMethod)
            """ lossType: hinge, msq, smx """
            zLastUpdateOpt = {'hinge': self.zLastUpdateWithHinge, 'msq': self.zLastUpdateWithMeanSq, 'smx': self.zLastUpdateWithSoftmax }
            z[L] = zLastUpdateOpt[lossType](beta, waL, y, Lambda, method= minMethod, tau=tau , ite=ite)

            # lambda update
            if hasLambda:
               Lambda += beta * (z[L] - waL)
            
            # Update beta, gamma
            beta *= 1
            gamma *= 1
            
            # Calculate total loss
            if calLoss:
                self.calLoss(beta, gamma, a, z, w, y, Lambda, lossType)

        return w, Lambda

    def train(self, train, weightConsWeight, activConsWeight, iterOutNum, iterInNum, hasLambda, calLoss=False, batchSize=0, lossType='smx', minMethod='prox', tau=0.01, ite=25, initW=None ):
        
        #Xtr, Ytr = train.images, train.labels
        Xtr, Ytr = train.nextBatch(batchSize)
        
        # Initialization 
        # - C: number of classes, N: number of training images, L: number of layers(including output layer)
        C = self.classNum
        N = batchSize
        L = len(self.hiddenLayer) + 1

        # - beta,gama: penalty coefficiencies
        beta = 1.0 * weightConsWeight 
        gamma = 1.0 * activConsWeight

        a, z, w = self.initNetwork(Xtr, C, self.hiddenLayer, self.epsilon, initW)   
        Lambda = np.zeros_like(z[L])
                       
        # Main part of ADMM updates
        for k in range(iterOutNum): 
            # Transform y to hotone representation
            y = self.toHotOne(Ytr, C)
            
            # ADMM Update
            #if k == iterOutNum-1:
             #   iterInNum *= 2 
            w, Lambda = self.admmUpdate(y, a, z, w, L, iterInNum, beta, gamma, hasLambda, calLoss, lossType, minMethod, tau, ite, Lambda)
            
            # Load new batch
            Xtr, Ytr = train.nextBatch(batchSize)
            # - a: activation, z: output, w: weight
            a, z, w = self.initNetwork(Xtr, C, self.hiddenLayer, self.epsilon, w)  
            #print self.dataLoss[len(self.dataLoss)-1]
            print np.max(w[L]) 
        
        self.W = w
        self.zL = z[L]
    
    def trainWithoutMiniBatch(self, weightConsWeight, activConsWeight, iterNum, hasLambda, calLoss=False, lossType='smx', minMethod='prox', tau=0.01, ite=25, initW=None ):
        # Initialization 
        # - C: number of classes, N: number of training images, L: number of layers(including output layer)
        C = self.classNum
        N = self.trainNum
        L = len(self.hiddenLayer) + 1
        
        # - beta,gama: penalty coefficiencies
        beta = 1.0 * weightConsWeight 
        gamma = 1.0 * activConsWeight

        C = self.classNum
        a, z, w = self.initNetwork(self.Xtr, C , self.hiddenLayer, self.epsilon, initW)   
        Lambda = np.zeros_like(z[L])
        y = self.toHotOne(self.Ytr, C)
      
        print 'train options:\nlossType:%s'%lossType
        print 'minMethod:%s tau:%f ite:%d'%( minMethod, tau, ite)

	# ADMM Update
        w, Lambda = self.admmUpdate(y, a, z, w, L, iterNum, beta, gamma, hasLambda, calLoss, lossType, minMethod, tau, ite, Lambda)
            
        # Save the W to network
        self.W = w
        self.zL = z[L]     
      
    def predict(self, Xte):
        """
        Inputs:
        - w: A list containing weight matrix of each layer
        - Xte: A numpy array containing all test images
        Return:
        - Yte: A numpy array containing predicted labels of input images
        """
        w = self.W
        z = w[1].dot(Xte)
        for l in range(1,len(self.hiddenLayer)+1):
            z = w[l+1].dot(self.ReLU(z))
        y = z
        return  np.argmax(y, axis=0), y

    def predictByFeed(self, Xte, w):
        
        z = w[1].dot(Xte)
        for l in range(1,len(self.hiddenLayer)+1):
            z = w[l+1].dot(self.ReLU(z))
        y = z
        return  np.argmax(y, axis=0), y

   
    def zUpdate(self, beta, gamma, wa, al):
        # z_i < 0
        z_s = np.copy(wa)
        z_s[z_s > 0] = 0
        loss_s = self.quadraCost(beta, gamma, al, wa, z_s) # !!

        # z_i > 0 
        z_b = (gamma * al + beta * wa) / (beta + gamma)
        z_b[z_b < 0] = 0
        loss_b = self.quadraCost(beta, gamma, al, wa, z_b)
        
        z_s[loss_s > loss_b] = z_b[loss_s > loss_b]
        
        return np.copy(z_s)

    def zLastUpdateWithSoftmax(self, beta, waL, y, Lambda, method=None, tau=None, ite=None):
        zL = np.zeros(waL.shape)
        if method == 'gd':
           zL = self.minZWithGD(beta, waL, y, Lambda, tau, ite)
           
        if method == 'prox':
           zL = self.minZWithProx(beta, waL, y, Lambda, tau, ite)

        if method == 'newton':
           zL = self.minZWithNewton(beta, waL, y, Lambda, tau, ite)
        return zL

    def minZWithNewton(self, beta, waL, y, Lambda, tau, ite):
        zL = np.copy(waL)  
        N = zL.shape[1] 
        
        for i in range(ite):
            # calculate probabilities
            #print zL            
            zExp = np.exp(zL) 
            zProb = 1.0 * zExp / np.sum(zExp, axis=0, keepdims=True)

            # calculate gradient of z
            dLdz = (zProb - y)  + 2 * beta * (zL - waL)  + Lambda
            diag = zProb*(1-zProb)
            H = np.zeros((zL.shape[0],zL.shape[0]))
            for i in range(0,zL.shape[1]):
                pi =  zProb[:,i].reshape(zL.shape[0],1);
                dLdzi = -1 * pi.dot(pi.T);
                np.fill_diagonal(dLdzi,diag[:,i])
                H += dLdzi
                
            H = H / N + 2 * beta * np.identity(zL.shape[0])

            # update
            zL = zL - tau * np.linalg.inv(H).dot(dLdz)
        return zL


    def minZWithProx(self, beta, waL, y, Lambda, tau, ite):
        zL = np.copy(waL) 
        N = zL.shape[1]         
        
        for i in range(ite):
            #zL = np.zeros(waL.shape)
            # calculate probabilities
            zExp = np.exp(zL) 
            zProb = 1.0 * zExp / np.sum(zExp, axis=0, keepdims=True)

            # calculate gradient of z
            dLdz = (zProb - y)
            v = zL - tau * (dLdz)

            # update
            zL = (2 * tau * beta * waL - tau * Lambda + v) / (1 + 2 * tau * beta)
        return zL



    def minZWithGD(self, beta, waL, y, Lambda, step, ite):
        #zL = np.zeros(waL.shape)
        zL = np.copy(waL)    
        # minimize with gradient descent
        for i in range(ite):
            # calculate probabilities
            zExp = np.exp(zL) 
            zProb = 1.0 * zExp / np.sum(zExp, axis=0, keepdims=True)

            # calculate gradient of z
            dLdz = zProb - y
            dEdz = dLdz + Lambda + 2 * beta * (zL - waL)

            # descent
            zL = zL - step * dEdz
        return zL


    def zLastUpdateWithHinge(self, beta, waL, y, Lambda, method=None, tau=None, ite=None):

        zL = np.zeros(waL.shape)
        # y_i = 1
        # zi > 1
        zL_b = np.copy(waL) - Lambda / (2 * beta)
        zL_b[zL_b < 1] = 1
        lossL_b = self.outputCost(beta, waL, zL_b, y, 1, Lambda)
        
        # zi < 1
        zL_s = waL +  (1 - Lambda) / (2 * beta)
        zL_s[zL_s > 1] = 1
        lossL_s = self.outputCost(beta, waL, zL_s, y, 1, Lambda)
        
        zL_s[lossL_s > lossL_b] = zL_b[lossL_s > lossL_b]
        zL[y == 1] = zL_s[y == 1]

        # y_i = 0
        # zi < 0
        zL_s = np.copy(waL) - Lambda / (2 * beta)
        zL_s[zL_s > 0] = 0
        lossL_s = self.outputCost(beta, waL, zL_s, y, 0, Lambda)

        # zi > 0
        zL_b = waL - (1 + Lambda) / (2 * beta)
        zL_b[zL_b < 0] = 0
        lossL_b = self.outputCost(beta, waL, zL_b, y, 0, Lambda)
            
        zL_s[lossL_s > lossL_b] = zL_b[lossL_s > lossL_b]
        zL[y == 0] = zL_s[y == 0] 
        return zL
    
    def zLastUpdateWithMeanSq(self, beta, waL, y, Lambda, method=None, tau=None, ite=None):
        return (y + beta * waL)/ (1 + beta) - 0.5 * Lambda / (1 + beta)

    def meanSqrLoss(self, z, y):
        return np.sum(z - y)

    def toHotOne(self, Y, C):
        """ Construct Hot-one representation of Y """
        y = np.zeros((C, Y.shape[0]))
        for i in range(0,Y.shape[0]):
            y[Y[i],i] = 1
        return y

    def softMaxLossTest(self, w):
        y = self.toHotOne(self.Ytr, self.classNum) 
        z = w[1].dot(self.Xtr)
        for l in range(1,len(self.hiddenLayer)+1):
            z = w[l+1].dot(self.ReLU(z))
        loss = np.sum(self.softMax(z, y)) / self.trainNum
        return loss

    def getFinalDataLoss(self, beta, gamma,lossType):
        C = self.classNum
        y = self.toHotOne(self.Ytr, C)
        L = len(self.hiddenLayer) + 1
        dataLossOpt = {'hinge': self.hinge, 'msq': self.meanSqr, 'smx': self.softMax}
        dataLoss = np.sum(dataLossOpt[lossType](self.zL, y)) / self.Xtr.shape[1]
        return dataLoss


    def calLoss(self, beta, gamma, a, z, w, y, Lambda, lossType):
        L = len(self.hiddenLayer) + 1
        TOTAL = 0
        # cal data cost, lossType: hinge, msq, smx
        dataLossOpt = {'hinge': self.hinge, 'msq': self.meanSqr, 'smx': self.softMax}
        dataLoss = np.sum(dataLossOpt[lossType](z[L], y)) / a[0].shape[1]
       
        self.dataLoss.append(dataLoss)
        TOTAL += dataLoss

        # cal lagrange cost 
        lagraLoss = np.sum(z[L] * Lambda)
        self.lagraLoss.append(lagraLoss)

        # cal quaratic cost 
        for l in range(1,L):
          aLoss = np.sum(self.aQuadraLoss(gamma, a[l], z[l]))
          self.aConstrLoss[l-1].append(aLoss)

          zLoss = np.sum(self.zQuadraLoss(beta, w[l].dot(a[l-1]), z[l]))
          self.zConstrLoss[l-1].append(zLoss)
          TOTAL += aLoss + zLoss

        zLastLoss = np.sum(self.zQuadraLoss(beta, w[L].dot(a[L-1]), z[L]))
        self.zConstrLoss[L-1].append(zLastLoss)
        TOTAL += zLastLoss
        self.totalLoss.append(TOTAL)    

   
    def outputCost(self, beta, wa, z, y, isOne, Lambda): # can be improved
        return self.hingeLoss(z, y, isOne) + self.zQuadraLoss(beta, wa, z) + z * Lambda

    def quadraCost(self, beta, gamma, a, wa, z):
        return  self.aQuadraLoss(gamma, a, z) + self.zQuadraLoss(beta, wa, z)

    def zQuadraLoss(self, beta, wa, z):
        return beta * (z - wa) ** 2
    
    def aQuadraLoss(self, gamma, a, z):
        return gamma * (a - self.ReLU(z)) ** 2 
    
    def ReLU(self, x):
        """ Evaluate Rectified Linear Unit """
        xn = np.copy(x)
        xn[xn < 0] = 0
        return xn

    def hinge(self, z, y): # to replace the old one
        loss = np.zeros(y.shape)
        loss[y == 0] = np.maximum(z,loss)[y == 0]
        loss[y == 1] = np.maximum(1-z,loss)[y == 1]
        return loss

    def meanSqr(self, z, y):
        return (z - y) ** 2
    
    def hingeLoss(self, z, y, isOne): # can be improved
        """ Evaluate Hinge Loss """
        zn = np.copy(z)
        x = np.zeros(z.shape)
        if not isOne:
            zn[y == 0] = np.maximum(zn,x)[y == 0]
        else:
            zn[y == 1] = np.maximum(1-zn,x)[y == 1]
        return zn

    def softMax(self, z, y):
        """ Evaluate Multiclass SVM Loss """ 
        loss = np.zeros(y.shape)
        zExp = np.exp(z) 
        zProb = 1.0 * zExp / np.sum(zExp, axis=0, keepdims=True) 
        loss = -1 * y * np.log(zProb)
        return loss
   

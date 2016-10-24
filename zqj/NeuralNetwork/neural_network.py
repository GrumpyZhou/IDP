import numpy as np
from random import shuffle

class NeuralNetwork():
    """
    A neural network with L layers, for each layer the dimension of neurons is specified in Dim[],
    Dim[0] is later filled by the input dimension.
    """
    def __init__(self, Xtr, Ytr, classNum, hiddenLayer, epsilon ):
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
        
        self.dataLoss = [] # cost from loss function
        self.aConstrLoss = [] # cost from constraint a = h(z)
        self.zConstrLoss = [] # cost from constraint z = wa
        self.lagraLoss = [] # cost from lagrange term
        for l in range(1,len(self.hiddenLayer)+ 1):
            self.aConstrLoss.append([])
            self.zConstrLoss.append([])
        self.zConstrLoss.append([]) # cost from output layer zL

        print self.dataLoss, ' ',self.aConstrLoss, ' ', self.zConstrLoss
        print "Initializing a neural network with : ", len(hiddenLayer)," hidden layers, hidden layer dimension:", hiddenLayer
        
        
    def initNetwork(self, trainNum, classNum, hiddenLayer, epsilon):
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

    def train(self, weightConsWeight, activConsWeight, iterNum, hasLambda, calLoss, lossType = 'smx', minMethod = 'prox', tau=0.01, ite= 25):

        # Initialization 
        # - C: number of classes, N: number of training images, L: number of layers(including output layer)
        C = self.classNum
        N = self.trainNum
        L = len(self.hiddenLayer) + 1
        
        # - beta,gama: penalty coefficiencies
        beta = 1.0 * weightConsWeight 
        gamma = 1.0 * activConsWeight

        # - a: activation, z: output, w: weight
        a, z, w = self.initNetwork(self.trainNum, self.classNum, self.hiddenLayer, self.epsilon)
        Lambda = np.zeros_like(z[L])
        
        # Transform y to hotone representation
        y = self.toHotOne(self.Ytr, C) 
               
        # Main part of ADMM updates
        for k in range(iterNum):
            # Walk through 1~L-1 layer network
            for l in range(1, L):
                # w update
                w[l] = np.linalg.lstsq(a[l-1].T,z[l].T)[0].T
                #w[l] = z[l].dot(np.linalg.pinv(a[l-1]))
               
                # a update
                wNtr = w[l+1].T
                a[l] = np.linalg.inv(beta * wNtr.dot(w[l+1]) + gamma * np.identity(wNtr.shape[0])).dot(beta * wNtr.dot(z[l+1]) + gamma * self.ReLU(z[l]))
               
                # z update
                z[l] = self.zUpdate(beta, gamma, w[l].dot(a[l-1]), a[l])
                           
            # L-layer
            # w update
            w[L] = np.linalg.lstsq(a[L-1].T,z[L].T)[0].T            
            #w[L] = z[L].dot(np.linalg.pinv(a[L-1]))
            
            # zL update
            waL =  w[L].dot(a[L-1])

            #print 'Train model: lossType %s, minMethod %s', (lossType, minMethod)
            """ lossType: hinge, msq, smx """
            zLastUpdateOpt = {'hinge': self.zLastUpdateWithHinge, 'msq': self.zLastUpdateWithMeanSq, 'smx': self.zLastUpdateWithSoftmax }
            #z[L] = zLastUpdateOpt[lossType](beta, waL, y, Lambda, method= None, tau=None, ite=None)
            z[L] = zLastUpdateOpt[lossType](beta, waL, y, Lambda, method= minMethod, tau=0.01 , ite= 25)
            

            # lambda update
            if hasLambda:
               Lambda += beta * (z[L] - waL)
            
            # Update beta, gamma
            beta *= 1
            gamma *= 1
            
            # Calculate total loss
            if calLoss:
                self.calLoss(beta, gamma, a, z, w, y, Lambda, lossType)

        # Save the W to network
        self.W = w

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
        return  np.argmax(y, axis=0)

   
    def zUpdate(self, beta, gamma, wa, al):
        # z_i < 0
        z_s = np.copy(wa)
        z_s[z_s > 0] = 0
        loss_s = self.quadraCost(beta, gamma, al, wa, z_s) # !!

        # z_i > 0 
        z_b = (gamma * al + beta * z_s) / (beta + gamma)
        z_b[z_b < 0] = 0
        loss_b = self.quadraCost(beta, gamma, al, wa, z_b)
        
        z_s[loss_s > loss_b] = z_b[loss_s > loss_b]
        
        return np.copy(z_s)

    def zLastUpdateWithSoftmax(self, beta, waL, y, Lambda, method=None, tau=None, ite=None):
        zL = np.zeros(waL.shape)
        if method == 'gd':
           zL = self.minZWithGD(beta, waL, y, Lambda, tau, ite)
           
        if method == 'prox':
           zL = self.minZwithProx(beta, waL, y, Lambda, tau, ite)
        return zL

    def minZwithProx(self, beta, waL, y, Lambda, tau, ite):
        zL = np.copy(waL)    
        
        for i in range(ite):
            #zL = np.zeros(waL.shape)
            # calculate probabilities
            zExp = np.exp(zL) 
            zProb = 1.0 * zExp / np.sum(zExp, axis=0, keepdims=True)

            # calculate gradient of z
            dLdz = zProb - y
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


    def calLoss(self, beta, gamma, a, z, w, y, Lambda, lossType):
        L = len(self.hiddenLayer) + 1
        # cal data cost, lossType: hinge, msq, smx
        dataLossOpt = {'hinge': self.hinge, 'msq': self.meanSqr, 'smx': self.softMax}
        dataLoss = np.sum(dataLossOpt[lossType](z[L], y)) / self.trainNum
        self.dataLoss.append(dataLoss)

        # cal lagrange cost 
        lagraLoss = np.sum(z[L] * Lambda)
        self.lagraLoss.append(lagraLoss)
        
        # cal quaratic cost 
        for l in range(1,L):
            aLoss = np.sum(self.aQuadraLoss(gamma, a[l], z[l]))
            self.aConstrLoss[l-1].append(aLoss)
            
            zLoss = np.sum(self.zQuadraLoss(beta, w[l].dot(a[l-1]), z[l]))
            self.zConstrLoss[l-1].append(zLoss)

        zLastLoss = np.sum(self.zQuadraLoss(beta, w[L].dot(a[L-1]), z[L]))
        self.zConstrLoss[L-1].append(zLastLoss)

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
   

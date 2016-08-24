import numpy as np
from random import shuffle

class NeuralNetwork():
    """
    A neural network with L layers, for each layer the dimension of neurons is specified in Dim[],
    Dim[0] is later filled by the input dimension.
    """
    def __init__(self, layers, neurDim):
        self.L = layers
        self.Dim = neurDim
        print "Initializing a neural network: ", self.L," layers; Neuron Dimension:", neurDim

    def train(self, W, Xtr, Ytr, weightConsWeight, activConsWeight, iterNum, epsilon):
        """
        Inputs have dimension D, there are C classes, and operate on minibatches
        of N examples.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - Xtr: A numpy array of shape (N, D) containing a minibatch of data.
        - Ytr: A numpy array of shape (N,) containing training labels;         
        Return:
        - w: A list containing weight matrix of each layer
        """
        C = W.shape[1]
        N = Xtr.shape[0]
        L = self.L
        Dim = self.Dim
        Dim[0] = Xtr.shape[1]
        epsilon = epsilon

        # Initialization 
        # - a[]: activation list for each layer [a0, a1, a2]: a0(N,Dim[0]), a1(N,Dim[1]),a2(N,Dim[2])
        # - z[]: z list for each layer [0, z1, z2, z3]: z1(N,Dim[1]),z2(N,Dim[2]), z3(N,C)
        # - w[]: weight list for each layer  [0, w1, w2, w3]: w1(Dim[0],Dim[1]), w2(Dim[1],Dim[2]), w3(Dim[2],C)

        a = [Xtr, epsilon*np.random.randn(N, Dim[1]), epsilon*np.random.randn(N, Dim[2])] # whether to vectorize
        z = [np.zeros((0)), epsilon*np.random.randn(N, Dim[1]), epsilon*np.random.randn(N, Dim[2]), epsilon*np.random.randn(N, C)]
        w = [np.zeros((0)), epsilon*np.random.randn(Dim[0], Dim[1]), epsilon*np.random.randn(Dim[1], Dim[2]), epsilon*np.random.randn(Dim[2], C)]


        # - beta,gama: penalty coefficiencies
        # - K: it  erations of ADMM
        beta = 1.0 * weightConsWeight 
        gamma = 1.0 * activConsWeight
        K = iterNum
        
        # Main part of ADMM updates
        for k in range(K):

            # Walk through L-layer network
            for l in range(1, L):
                w[l] = np.linalg.pinv(a[l-1]).dot(z[l])
                wNtr = w[l+1].T
                a[l] = (beta * z[l+1].dot(wNtr) + gamma * self.neuronReLU(z[l])).dot(np.linalg.inv(beta * (w[l+1].dot(wNtr) + gamma * np.identity(wNtr.shape[1]))))
                                
                # z update
                aw = a[l-1].dot(w[l])
                
                # z_i < 0
                z_s = np.copy(aw)
                z_s[z_s > 0] = 0
                l_s = self.regularElementWiseCost(beta, gamma, a[l], aw, z_s)

                # z_i > 0 erer
                z_b = (gamma * a[l] + beta * z_s) / (beta + gamma)
                z_b[z_b < 0] = 0
                l_b = self.regularElementWiseCost(beta, gamma, a[l], aw, z_b)
                
                z_s[l_s > l_b] = z_b[l_s > l_b]
                z[l] = np.copy(z_s)
              
              
            # L-layer
            w[L] = np.linalg.pinv(a[L-1]).dot(z[L])
            

            # Transform y to hotone representation
            y = self.toHotOne(Ytr, C)

            # Hinge Loss
            awL = a[L-1].dot(w[L])
            zL = np.zeros(awL.shape)

            # y_i = 1
            # zi > 1
            zL_b = np.copy(awL)
            zL_b[zL_b < 1] = 1
            lL_b = self.outputElementWiseCost(beta, awL, zL_b, y, 1)
            
            # zi < 1
            zL_s = np.copy(awL + 1 / (2 * beta))
            zL_s[zL_s > 1] = 1
            lL_s = self.outputElementWiseCost(beta, awL, zL_s, y, 1)
            
            zL_s[lL_s > lL_b] = zL_b[lL_s > lL_b]
            zL[y == 1] = zL_s[y == 1]

            # y_i = 0
            # zi < 0
            zL_s = np.copy(awL)
            zL_s[zL_s > 0] = 0
            lL_s = self.outputElementWiseCost(beta, awL, zL_s, y, 0)

            # zi > 0
            zL_b = np.copy(awL - 1 / (2 * beta))
            zL_b[zL_b < 0] = 0
            lL_b = self.outputElementWiseCost(beta, awL, zL_b, y, 0)
                
            zL_s[lL_s > lL_b] = zL_b[lL_s > lL_b]
            zL[y == 0] = zL_s[y == 0]
            
            # Update zL
            z[L] = zL
            
            # Update beta, gamma
            beta *= 1.05
            gamma *= 1.05
            
            # Calculate loss
            #loss = calcuLoss(w, a, z, Ytr, gamma, beta)
            #print "Loss of iter ",k,":", loss

        return w

    def predict(self, w, Xte):
        """
        Inputs:
        - w: A list containing weight matrix of each layer
        - Xte: A numpy array containing all test images
        Return:
        - Yte: A numpy array containing predicted labels of input images
        """
        y = self.neuronReLU(self.neuronReLU(Xte.dot(w[1])).dot(w[2])).dot(w[3])
        return  np.argmax(y, axis=1)

    def calcuLoss(self, w, a, z, y, gamma, beta):
        
        # Hinge Loss / multi SVM Loss
        hingeLoss = self.hingeLoss(z[L], y)
        for l in range(1,L):
            regLoss +=  self.regularCost(beta, gamma, w[l], a[l], a[l-1], z_b)

        totalLoss = hingeLoss + beta * np.linalg.norm(z[L] - a[L-1].dot(w[L])) ** 2 + regLoss 
        return loss

    def toHotOne(self, Y, C):
        """ Construct Hot-one representation of Y """
        y = np.zeros((Y.shape[0],C))
        for i in range(0,Y.shape[0]):
            y[i, Y[i]] = 1
        return y

    def outputElementWiseCost(self, beta, aw, z, y, isOne):
        return self.hingeLossElementWiseCost(z,y, isOne) + beta * (z - aw) ** 2 

    def regularElementWiseCost(self, beta, gamma, a, aw, z):
        """ Calculate elementwise cost """
        return  gamma * (a - self.neuronReLU(z)) ** 2 + beta * (z - aw) ** 2 
            
    def neuronReLU(self, x):
        """ Evaluate Rectified Linear Unit """
        xn = np.copy(x)
        xn[xn < 0] = 0
        return xn

    def hingeLossElementWiseCost(self, z, y, isOne):
        """ Evaluate Hinge Loss """
        zn = np.copy(z)
        x = np.zeros(z.shape)
        if not isOne:
            zn[y == 0] = np.maximum(zn,x)[y == 0]
        else:
            zn[y == 1] = np.maximum(1-zn,x)[y == 1]
        return zn
      
    def multiSVMLoss(self, Spre,Ytrue):
        """ Evaluate Multiclass SVM Loss """        
        score = np.empty(zL.shape)
        for j in range(N):
            yj = Ytrue[j] # label of class
            S_yj = z[yj] # Score of jth class
            score[j].fill(S_yj - 1)
            
        delta = Spre - score
        delta[ delta < 0] = 0
        # sum loss along all pixels and along all pictures / N
        return np.sum(np.sum(delta, axis = 1), axis = 0) / Ytrue.shape[0]   

    def regularCost(self, beta, gamma, w, a, aPre, z):
        return  gamma * np.linalg.norm(a - self.neuronReLU(z)) ** 2 + beta * np.linalg.norm(z - aPre.dot(w)) ** 2 
 
   

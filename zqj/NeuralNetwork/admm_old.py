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
        print "initialize neural networks with ", self.L,"; Dim:", neurDim

    def train(self, W, Xtr, Ytr, weightConsWeight, activConsWeight, iterNum):
        """
        Inputs have dimension D, there are C classes, and operate on minibatches
        of N examples.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - Xtr: A numpy array of shape (N, D) containing a minibatch of data.
        - Ytr: A numpy array of shape (N,) containing training labels; Ytr[i] = c means
        that Xtr[i] has label c, where 0 <= c < C.
        """
        C = W.shape[1]
        N = Xtr.shape[0]
        L = self.L
        Dim = self.Dim
        Dim[0] = Xtr.shape[1]


        # For every input 
        # for i in xrange(trainNum):

        # Initialization 
        # - a[]: activation list for each layer [a0, a1, a2]: a0(N,Dim[0]), a1(N,Dim[1]),a2(N,Dim[2])
        # - z[]: z list for each layer [0, z1, z2, z3]: z1(N,Dim[1]),z2(N,Dim[2]), z3(N,C)
        # - w[]: weight list for each layer  [0, w1, w2, w3]: w1(Dim[0],Dim[1]), w2(Dim[1],Dim[2]), w3(Dim[2],C)

        a = [Xtr, np.zeros((N, Dim[1])), np.zeros((N, Dim[2]))] # whether to vectorize
        z = [0, np.zeros((N, Dim[1])), np.zeros((N, Dim[2])), np.zeros((N, C))]
        w = [0, np.zeros((Dim[0], Dim[1])), np.zeros((Dim[1], Dim[2])), np.zeros((Dim[2], C))]

        # - beta,gama: penalty coefficiencies
        # - K: it  erations of ADMM
        beta = weightConsWeight
        gamma = activConsWeight
        K = iterNum
        
        # Main part of ADMM updates
        for k in range(K):

            # Walk through L-layer network
            for l in range(1, L):
                w[l] = np.linalg.pinv(a[l-1]).dot(z[l])
                wNTr = w[l+1].T
                a[l] = np.linalg.inv(beta * (wNtr.dot(w[l+1]) + gamma) * (beta * z[l+1].dot(wNTr)) + gamma * neuronReLU(z[l]))

                # z update 
                # z_i < 0
                z_s = a[l-1].dot(w[l])
                z_s[z_s>0] = 0
                l_s = regularCost(beta, gamma, w[l], a[l], a[l-1], z_s)

                # z_i > 0
                z_b = (gamma * a[l] + beta * z_s) / (beta + gamma)
                z_b[z_b<0] = 0
                l_b =regularCost(beta, gamma, w[l], a[l], a[l-1], z_b)

                if l_s <= l_b:
                    z[l] = z_s
                else:
                    z[l] = z_b

            # L-layer
            w[L] = np.linalg.pinv(a[L-1]).dot(z[L])
            
            # Multiclass SVMs Loss
            zL_s = a[L-1].dot(w[L])
            zL_b = zL_s -1 / (2 * beta) # * N ??

            # Get the predicted score matrix of correct label
            score = np.empty((N,C))
            for j in range(N):
                yj = Ytr[j] # label of class
                S_yj = z[yj] # Score of jth class
                score[j].fill(S_yj - 1)
            
            zL = np.copy(zL_s)
            delta = zL_s - score
            zL[delta > 0] = zL_b[delta > 0] 
            z[L] = zL
            
            # Update beta, gamma
            beta *= 1.05
            gamma *= 1.05
            
            # Calculate loss
            
        return w


        
    def regularCost(beta, gamma, w, a, aPre, z):
        return   gamma * np.linalg.norm(a - neuronReLU(z)) ** 2 + beta * np.linalg.norm(z - aPre.dot(w)) ** 2 
        
    def neuronReLU(self, x):
        xn = np.copy(x)
        xn[xn < 0] = 0
        return xn

    def multiSVMLoss(Spre,Ytrue):
        
        score = np.empty(zL.shape)
        for j in range(N):
            yj = Ytrue[j] # label of class
            S_yj = z[yj] # Score of jth class
            score[j].fill(S_yj - 1)
            
        delta = Spre - score
        delta[ delta < 0] = 0

        return np.sum(np.sum(delta, axis = 1), axis = 0) / Ytrue.shape[0]   # sum loss along all pixels and along all pictures / N
    
    def calcuLoss(self, w, a, z):
        loss = 0.0
        
        loss = 

        return loss

    def check(Ypr,Yte):


        return loss


    def predict(W,Xte):


        return Y

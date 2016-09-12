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
        self.g1 = [0]
        self.g2 = [0]
        self.b1 = [0]
        self.b2 = [0]
        self.b3 = [0]
        self.lossL = [0]
        #print "Initializing a neural network with : ", len(hiddenLayer)," hidden layers, hidden layer dimension:", hiddenLayer
        
        
    def initNetwork(self, trainNum, classNum, hiddenLayer, epsilon):
        """ 
        Return:
        - A: Activation list for each layer [a0, a1, a2]
        - Z: A z list for each layer [0, z1, z2, z3]
        - W: weight list for each layer  [0, w1, w2, w3]
        """
        A = [self.Xtr, epsilon*np.random.randn(hiddenLayer[0], trainNum), epsilon*np.random.randn(hiddenLayer[1], trainNum)] 
        Z = [np.zeros((0)), epsilon*np.random.randn(hiddenLayer[0], trainNum), epsilon*np.random.randn(hiddenLayer[1], trainNum), epsilon*np.random.randn(classNum, trainNum)]
        W = [np.zeros((0)), epsilon*np.random.randn(hiddenLayer[0], self.Xtr.shape[0]), epsilon*np.random.randn(hiddenLayer[1], hiddenLayer[0]), epsilon*np.random.randn(classNum,hiddenLayer[1])]
         
        return A, Z, W


    def train(self, weightConsWeight, activConsWeight, iterNum, hasLambda):

        C = self.classNum
        N = self.trainNum
        L = len(self.hiddenLayer) + 1
        a, z, w = self.initNetwork(self.trainNum, self.classNum, self.hiddenLayer, self.epsilon)

        #a = [self.Xtr]
        #z = [0, w[1].dot(self.Xtr) ]

        #a.append(self.ReLU(z[1]))
        #z.append(w[2].dot(a[1]))

        #a.append(self.ReLU(z[2]))
        #z.append(w[3].dot(a[2]))

        Lambda = np.zeros_like(z[L])
        y = self.toHotOne(self.Ytr, C)  # Transform y to hotone representation

        # - beta,gama: penalty coefficiencies
        # - K: it  erations of ADMM
        beta = 1.0 * weightConsWeight 
        gamma = 1.0 * activConsWeight
        
        loss = [0,0,0,0]
        # Main part of ADMM updates
        for k in range(iterNum):

            # Walk through 1~L-1 layer network
            for l in range(1, L):
                # w update solve ||Wa-z||^2
                #w[l] = np.linalg.lstsq(a[l-1].T,z[l].T)[0].T
                w[l] = z[l].dot(np.linalg.pinv(a[l-1]))
               
                # a update
                wNtr = w[l+1].T
                a[l] = np.linalg.inv(beta * wNtr.dot(w[l+1]) + gamma * np.identity(wNtr.shape[0])).dot(beta * wNtr.dot(z[l+1]) + gamma * self.ReLU(z[l]))
               
                # z update
                wa = w[l].dot(a[l-1])
                
                # z_i < 0
                z_s = np.copy(wa)
                z_s[z_s > 0] = 0
                loss_s = self.quadraCost(beta, gamma, a[l], wa, z_s) # !!

                # z_i > 0 
                z_b = (gamma * a[l] + beta * z_s) / (beta + gamma)
                z_b[z_b < 0] = 0
                loss_b = self.quadraCost(beta, gamma, a[l], wa, z_b)
                
                z_s[loss_s > loss_b] = z_b[loss_s > loss_b]
                z[l] = np.copy(z_s)            
            
            # L-layer
            # w update
            #w[L] = np.linalg.lstsq(a[L-1].T,z[L].T)[0].T            
            w[L] = z[L].dot(np.linalg.pinv(a[L-1]))
            
            # z update 
            waL = w[L].dot(a[L-1])
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
            
            # Update zL
            z[L] = zL

            # lambda update
            if hasLambda:
               Lambda += beta * (z[L] - waL)
            
            # Update beta, gamma
            beta *= 1.05
            gamma *= 1.05
            
            """ DEBUG: loss
            self.g1.append(np.sum(gamma * (a[1] - self.ReLU(z[1])) ** 2))
            self.g2.append(np.sum(gamma * (a[2] - self.ReLU(z[2])) ** 2))
            self.b1.append(np.sum(beta * (z[1] - w[1].dot(a[0])) ** 2))
            self.b2.append(np.sum(beta * (z[2] - w[2].dot(a[1])) ** 2))
            self.b3.append(np.sum(beta * (z[L] - w[L].dot(a[L-1])) ** 2))
            """
            
            
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
        y = w[3].dot(self.ReLU(w[2].dot(self.ReLU(w[1].dot(Xte)))))
        #print y[:, xrange(2)]
        return  np.argmax(y, axis=0)


    def toHotOne(self, Y, C):
        """ Construct Hot-one representation of Y """
        y = np.zeros((C, Y.shape[0]))
        for i in range(0,Y.shape[0]):
            y[Y[i],i] = 1
        return y


    def outputCost(self, beta, wa, z, y, isOne, Lambda):
        return self.hingeLoss(z, y, isOne) + beta * (z - wa) ** 2 + z * Lambda


    def quadraCost(self, beta, gamma, a, wa, z):
        """ Calculate elementwise cost """
        return  gamma * (a - self.ReLU(z)) ** 2 + beta * (z - wa) ** 2 
            
    
    def ReLU(self, x):
        """ Evaluate Rectified Linear Unit """
        xn = np.copy(x)
        xn[xn < 0] = 0
        return xn


    def hingeLoss(self, z, y, isOne):
        """ Evaluate Hinge Loss """
        zn = np.copy(z)
        x = np.zeros(z.shape)
        if not isOne:
            zn[y == 0] = np.maximum(zn,x)[y == 0]
        else:
            zn[y == 1] = np.maximum(1-zn,x)[y == 1]

        self.lossL.append(np.sum(zn))
        return zn


    def toEvenOddLabel(self, Y, C):
        return Y % 2
    
    def toBinary(self, Y):
        yb = Y % 2
        return self.toHotOne(yb, 2)
    
    def multiSVMLoss(self, Spre,Ytrue):
        """ Evaluate Multiclass SVM Loss """        
        pass  
   

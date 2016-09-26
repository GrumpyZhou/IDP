import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt
from numpy.linalg import lapack_lite

lapack_routine = lapack_lite.dgesv

def faster_inverse(A,idmatrix):
    b = idmatrix

    n_eq = A.shape[0]
    n_rhs = A.shape[1]
    pivots = zeros(n_eq, np.intc)
    identity  = np.eye(n_eq)

    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = zeros(n_eq, np.intc)
        results = lapack_lite.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        return b

    return array(lapack_inverse(A))

class NeuralNetwork():

    def __init__(self, layers, neurons, trainnum, testnum, iternum, stepsize, epsilon):

        '''
        :param layers: how many layers the network has
        :param neurons: the number of neurons in each layer
        :param trainnum: the number of training images
        :param testnum: the number of test images
        :param iternum: the iteration number for ADMM
        :param stepsize: the stepsize for BP
        :param epsilon: the constant for initializing w
        '''
        self.L = layers
        self.neurons = neurons
        self.trainnum = trainnum
        self.itnum = iternum
        self.testnum = testnum
        self.stepsize= stepsize
        self.beta = 1
        self.gamma = 10
        self.losscost = list()
        self.act1 = list()
        self.act2 = list()
        self.w1 = list()
        self.w2 = list()
        self.w3 = list()
       # self.grada = [0,0,0]
       # self.gradz = [0,0,0,0]
       # self.gradw = [0,0,0,0]

        '''
        initialize w, a, z
        '''
        a0, self.labels, self.y = self.loadmnist()
       # self.a = [a0,epsilon*np.random.randn(neurons[1],self.trainnum),epsilon*np.random.randn(neurons[2],self.trainnum)]
       # self.z = [0,epsilon*np.random.randn(neurons[1],self.trainnum),epsilon*np.random.randn(neurons[2],self.trainnum),epsilon*np.random.randn(neurons[3],self.trainnum)]
        self.w = [0,epsilon*np.random.randn(neurons[1],784),epsilon*np.random.randn(neurons[2],neurons[1]),epsilon*np.random.randn(neurons[3],neurons[2])]
        self.a = [a0]
        self.z = [0]
        self.z.append(self.w[1].dot(self.a[0]))
        self.a.append(self.actfun(self.z[1]))
        self.z.append(self.w[2].dot(self.a[1]))
        self.a.append(self.actfun(self.z[2]))
        self.z.append(self.w[3].dot(self.a[2]))

       # print "create a neural network with ", self.L, "layers. The number of neurons: ", self.neurons
       # print "w:", self.w[1].shape, self.w[2].shape, self.w[3].shape
       # print "z:", self.z[1].shape, self.z[2].shape, self.z[3].shape
       # print "a:", self.a[0].shape, self.a[1].shape, self.a[2].shape

    def loadmnist(self, dataset="training", digits=np.arange(10), path="./"):

        '''
        load the mnist data
        '''
        if dataset == "training":
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
            N = self.trainnum
        elif dataset == "testing":
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
            N = self.testnum
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        flbl = open(fname_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()
        ind = [k for k in range(size) if lbl[k] in digits]

        images = np.zeros((N, rows, cols), dtype=uint8)
        labels = np.zeros((N, 1), dtype=int8)
        for i in range(N):
            images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        re_img = np.zeros((784,N),dtype=np.float64)
        for i in range(N):
            re_img[:,i] = images[i,:,:].reshape(784)

        y = np.zeros((10,N),dtype=uint8)
        for i in range(len(labels)):
            y[labels[i],i] = 1
        #print sum(sum(y))
        return re_img, labels, y

    def trainByADMM(self):
        idmatrixes = list()
        idmatrixes.append(np.identity(self.w[2].T.shape[0]))
        idmatrixes.append(np.identity(self.w[3].T.shape[0]))
        idmatrixes.append(np.identity(self.trainnum))
        for k in range(self.itnum):
            #print k
            for i in range(1,self.L):
                self.w[i] = self.updateW(i,idmatrixes)
                self.a[i] = self.updateA(i,idmatrixes)
                self.z[i] = self.updateZ(i)

            self.w[self.L] = self.updateW(self.L,idmatrixes)
            self.z[self.L] = self.updateLastZ()
            self.beta *= 1.05
            self.gamma *= 1.05
            self.calEnergy()

    def updateW(self, i,idmatrixes):
        aT = self.a[i-1].T
        pinva = faster_inverse(aT.dot(self.a[i-1]),idmatrixes[2]).dot(aT)
        return self.z[i].dot(pinva)

    def updateA(self,i,idmatrixes):
        wT = self.w[i+1].T
        #tmp = np.linalg.solve(self.beta * (wT.dot(self.w[i+1]) + self.gamma*idmatrixes[i-1]), idmatrixes[i-1])
        tmp = faster_inverse(self.beta * (wT.dot(self.w[i+1]) + self.gamma*idmatrixes[i-1]), idmatrixes[i-1])
        return tmp.dot(self.beta * wT.dot(self.z[i + 1]) + self.gamma * self.actfun(self.z[i]))

    def updateZ(self,i):
        aw = self.w[i].dot(self.a[i - 1])

        z_s = np.copy(aw)
        z_s[z_s > 0] = 0
        l_s = self.regularElementWiseCost(self.beta,self.gamma, self.a[i], aw, z_s)

        z_b = (self.gamma * self.a[i] + self.beta * z_s) / (self.beta + self.gamma)
        z_b[z_b < 0] = 0
        l_b = self.regularElementWiseCost(self.beta, self.gamma, self.a[i], aw, z_b)

        z_s[l_s > l_b] = z_b[l_s > l_b]

        return z_s

    def updateLastZ(self):
        awL = self.w[self.L].dot(self.a[self.L-1])
        zL = np.zeros(awL.shape)

        zL_b = np.copy(awL)
        zL_b[zL_b < 1] = 1
        lL_b = self.outputElementWiseCost(self.beta, awL, zL_b, self.y, 1)

        zL_s = np.copy(awL + 1 / (2 * self.beta))
        zL_s[zL_s > 1] = 1
        lL_s = self.outputElementWiseCost(self.beta, awL, zL_s, self.y, 1)

        zL_s[lL_s > lL_b] = zL_b[lL_s > lL_b]
        zL[self.y == 1] = zL_s[self.y == 1]

        zL_s = np.copy(awL)
        zL_s[zL_s > 0] = 0
        lL_s = self.outputElementWiseCost(self.beta, awL, zL_s, self.y, 0)

        zL_b = np.copy(awL - 1 / (2 * self.beta))
        zL_b[zL_b < 0] = 0
        lL_b = self.outputElementWiseCost(self.beta, awL, zL_b, self.y, 0)

        zL_s[lL_s > lL_b] = zL_b[lL_s > lL_b]
        zL[self.y == 0] = zL_s[self.y == 0]

        return zL

    def hingeLossElementWiseCost(self, z, y, isOne):
        """ Evaluate Hinge Loss """
        zn = np.copy(z)
        x = np.zeros(z.shape)
        if not isOne:
            zn[y == 0] = np.maximum(zn, x)[y == 0]
        else:
            zn[y == 1] = np.maximum(1 - zn, x)[y == 1]
        return zn

    def outputElementWiseCost(self, beta, aw, z, y, isOne):
        return self.hingeLossElementWiseCost(z, y, isOne) + beta * (z - aw) ** 2

    def actfun(self,z):
        xn = np.copy(z)
        xn[xn < 0] = 0
        return xn

    def regularElementWiseCost(self, beta, gamma, a, aw, z):
        """ Calculate elementwise cost """
        return gamma * (a - self.actfun(z)) ** 2 + beta * (z - aw) ** 2

    def getY(self,a3):
        m, n = a3.shape
        y = zeros((n, 1), dtype=int)
        for i in range(n):
            y[i] = np.argmax(a3[:, i])
        return y

    def predict(self):
        a0,labels,y = self.loadmnist("training")
        result = self.w[3].dot(self.actfun(self.w[2].dot(self.actfun(self.w[1].dot(a0)))))
        result_labels = self.getY(result)
        count = 0
        # print(result_labels)
        # print(labels)
        for i in range(len(labels)) :
            if(result_labels[i]==labels[i]):
                count += 1
        print count*1.0/self.testnum

    def getW(self):
        return self.w

    def lossfun(self):
        tmp = np.zeros((10,self.trainnum))
        z = self.w[3].dot(self.actfun(self.w[2].dot(self.actfun(self.w[1].dot(self.a[0])))))
        x = np.zeros(z.shape)
        tmp[self.y==1] = np.maximum(1-z,x)[self.y==1]
        tmp[self.y==0] = np.maximum(z,x)[self.y==0]

    def calEnergy(self):
        self.losscost.append(self.lossfun())
        self.w1.append(sum(sum((self.z[1]-self.w[1].dot(self.a[0]))**2)))
        self.w2.append(sum(sum((self.z[2]-self.w[2].dot(self.a[1]))**2)))
        self.w3.append(sum(sum((self.z[3]-self.w[3].dot(self.a[2]))**2)))
        self.act2.append(sum(sum((self.a[2]-self.actfun(self.z[2]))**2)))
        self.act1.append(sum(sum((self.a[1]-self.actfun(self.z[1]))**2)))
        return 0

    def getEnergy(self):
        return self.w1,self.w2,self.w3,self.act1,self.act2

neurons = [784,100,150,10]
trainnum = 5000
testnum = 500
itnum = 30
stepsize = 0.1
nn = NeuralNetwork(3,neurons,trainnum,testnum,itnum,stepsize,0.01)
nn.trainByADMM()
nn.predict()
w1,w2,w3,act1,act2 = nn.getEnergy()
print w3

'''
plt.figure(1)
plt.plot(w1)
plt.figure(2)
plt.plot(w2)
plt.figure(3)
plt.plot(w3)
plt.figure(4)
plt.plot(act1)
plt.figure(5)
plt.plot(act2)
'''
plt.show()

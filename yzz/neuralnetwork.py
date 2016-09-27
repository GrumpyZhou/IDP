import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt
from numpy.linalg import lapack_lite
'''
lapack_routine = lapack_lite.dgesv

def faster_inverse(A,idmatrix):
	b = idmatrix

	n_eq = A.shape[0]
	n_rhs = A.shape[1]
	pivots = zeros(n_eq,np.intc)
	identity = np.eye(n_eq)

	def lapack_inverse(a):
		b = np.copy(identity)
		pivots = zeros(n_eq, np.intc)
		results = lapack_lite.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
		return b

	return array(lapack_inverse(A))
'''
class NeuralNetwork():

	def __init__(self, layers, neurons, trainnum, testnum, iternum, epsilon):
		self.L = layers
		self.neurons = neurons
		self.trainnum = trainnum
		self.itnum = iternum
		self.testnum = testnum
		self.beta = 1.0
		self.gamma = 10.0 
		self.lossType = "Mean Square"
		self.actType = "ReLu"
		self.losscost = list()
		self.act1 = list()
		self.act2 = list()
		self.w1 = list()
		self.w2 = list()
		self.w3 = list()

		a0, self.labels, self.y = self.loadmnist()

		self.w = [0]
		for i in range(1,self.L+1):
			self.w.append(epsilon*np.random.randn(neurons[i],neurons[i-1]))

		self.a = [a0]
		self.z = [0]
		for i in range(1,self.L):
			self.z.append(self.w[i].dot(self.a[i-1]))
			self.a.append(self.actFun(self.z[i]))

		self.z.append(self.w[self.L].dot(self.a[self.L-1]))
		self.mylambda = np.zeros_like(self.z[self.L],dtype=np.float64)

	def loadmnist(self, dataset="training", digits=np.arange(10), path="./"):
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
		re_img = re_img/255;

		y = np.zeros((10,N),dtype=uint8)
		for i in range(len(labels)):
			y[labels[i],i] = 1
        #print sum(sum(y))
		return re_img, labels, y

	def train(self):
		idmatrixes = list()
		for i in range(2,self.L+1):
			idmatrixes.append(np.identity(self.w[i].T.shape[0]))

		idmatrixes.append(np.identity(self.trainnum))

		for k in range(self.itnum):
			for i in range(1,self.L):
				self.w[i] = self.updateW(i,idmatrixes)
				self.a[i] = self.updateA(i,idmatrixes)
				self.z[i] = self.updateZ(i)

			self.w[self.L] = self.updateW(self.L, idmatrixes)
			#self.z[self.L] = 
			self.updateLastZ()
			self.mylambda += self.beta*(self.z[self.L]-self.w[self.L].dot(self.a[self.L-1]))
			#print self.mylambda
			#print self.mylambda.shape
			self.beta *= 1.05
			self.gamma *= 1.05
			self.calEnergy()

	#	print self.mylambda
#		print '-------------------------'
#		print self.z[self.L]

	def updateW(self, i, idmatrixes):
		#aT = self.a[i-1].T
		#pinva = faster_inverse(aT.dot(self.a[i-1]),idmatrixes[i-1]).dot(aT)
		#pinva = np.linalg.pinv(self.a[i-1])
		#pinva = np.linalg.lstsq(self.
		return np.linalg.lstsq(self.a[i-1].T,self.z[i].T)[0].T 

	def updateA(self, i, idmatrixes):
		wT = self.w[i+1].T
		tmp = np.linalg.inv(self.beta*(wT.dot(self.w[i+1])+self.gamma*idmatrixes[i-1]))
		return tmp.dot(self.beta*wT.dot(self.z[i+1])+self.gamma*self.actFun(self.z[i]))

	def updateZ(self,i):
		aw = self.w[i].dot(self.a[i - 1])
		z_s = np.copy(aw)
		z_s[z_s > 0] = 0
		l_s = self.regularElementWiseCost(self.beta,self.gamma, self.a[i], aw, z_s)

		z_b = (self.gamma * self.a[i] + self.beta * z_s) / (self.beta + self.gamma)
		z_b[z_b < 0] = 0
		l_b = self.regularElementWiseCost(self.beta, self.gamma, self.a[i],aw,z_b)
		z_s[l_s > l_b] = z_b[l_s > l_b]
		return z_s

	def updateLastZ(self):
		if self.lossType == "Mean Square":
			self.z[self.L] = (self.y*1.0+self.beta*self.w[self.L].dot(self.a[self.L-1])-self.mylambda*1.0/2)*1.0/(1+self.beta)
		elif self.lossType == "Softmax":
			#print 1
			self.z[self.L] = (2*self.beta*self.w[self.L].dot(self.a[self.L-1])-self.mylambda)/(1+2*self.beta)
			self.z[self.L][self.y==1] += 1 / (1+2*self.beta)
		return 0

	def hingeLossElementWiseCost(self,z,y,isOne):
		zn = np.copy(z)
		x = np.zeros(z.shape)
		if not isOne:
			zn[y==0] = np.maximum(zn,x)[y==0]
		else:
			zn[y==1] = np.maximum(1-zn,x)[y==1]
		return zn
	
	def outputElementWiseCost(self, beta,aw,z,y,isOne):
		return self.hingeLossElementWiseCost(z,y,isOne)+beta*(z-aw)**2

	def actFun(self,z):
		if self.actType=="ReLu":
			xn = np.copy(z)
			xn[xn<0] = 0
			return xn

	def regularElementWiseCost(self,beta,gamma,a,aw,z):
		return gamma*(a-self.actFun(z))**2 + beta*(z-aw)**2

	def getY(self,a3):
		m,n = a3.shape
		y = zeros((n,1),dtype=int)
		for i in range(n):
			y[i] = np.argmax(a3[:,i])
		return y

	def predict(self):
		a0,labels,y = self.loadmnist("training")
		a = list()
		z = list()
		a.append(a0)
		z.append(0)
		for i in range(1,self.L):
			z.append(self.w[i].dot(a[i-1]))
			a.append(self.actFun(z[i]))
		
		result = self.w[self.L].dot(a[self.L-1])
		#print result
		result_labels = self.getY(result)
		count = 0
		for i in range(len(labels)):
			if(result_labels[i] == labels[i]):
				count+=1
		print count*1.0/self.testnum

	def lossfun(self):
		if self.lossType == "Mean Square":
			return np.sum((self.z[self.L]-self.y)**2)*1.0/self.trainnum
		elif self.lossType == "Softmax":
			return np.sum(np.exp(self.z[self.L][self.y==1])/np.sum(np.exp(self.z[self.L]),axis=0))/self.trainnum
	
	def calEnergy(self):
		self.losscost.append(self.lossfun())
		self.w1.append(sum(sum((self.z[1]-self.w[1].dot(self.a[0]))**2)))
		self.w2.append(sum(sum((self.z[2]-self.w[2].dot(self.a[1]))**2)))
		self.act1.append(sum(sum((self.a[1]-self.actFun(self.z[1]))**2)))

	def getEnergy(self):
		return self.w1,self.w2,self.act1,self.losscost

neurons = [784,200,300,10]
trainnum = 10
testnum = 10
itnum = 30
stepsize = 0.1
epsilon = 0.00000001
nn = NeuralNetwork(3,neurons,trainnum,testnum,itnum,epsilon)
nn.train()
nn.predict()
w1,w2,act1,losscost = nn.getEnergy();

plt.figure(1)
plt.plot(w1)
plt.figure(2)
plt.plot(w2)
plt.figure(3)
plt.plot(act1)
plt.figure(4)
plt.plot(losscost)
plt.show()

import numpy as np
import os,struct
from numpy import append, array ,int8, uint8, zeros
from array import array as pyarray
import matplotlib.pyplot as plt

class NeuralNetwork():
	
	def __init__(self,layers, neurons, trainnum, testnum, iternum, stepsize, epsilon):
		self.L = layers
		self.neurons = neurons
		self.trainnum = trainnum
		self.testnum = testnum
		self.itnum = iternum
		self.stepsize = stepsize
		self.loss = list()
		a0, self.labels, self.y = self.loadmnist()

		self.w = [0,epsilon*np.random.randn(neurons[1],784),epsilon*np.random.randn(neurons[2],neurons[1]),epsilon*np.random.randn(neurons[3],neurons[2])]
		self.a = [a0,0,0]
		self.z = [0,0,0,0]
		
		self.gw = [np.array(0),np.array(0),np.array(0),np.array(0)]
		self.gz = [np.array(0),np.array(0),np.array(0),np.array(0)]
		self.ga = [np.array(0),np.array(0),np.array(0),np.array(0)]
		self.delta = [np.array(0),np.array(0),np.array(0),np.array(0)]

		loss = list()

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

		y = np.zeros((10,N),dtype=uint8)
		for i in range(len(labels)):
			y[labels[i],i] = 1
		return re_img, labels, y

	def train(self):
		for i in range(self.itnum):
			self.forePropagation()
			self.backPropagation()
			#self.gradientCheck()
			#print self.gw[1]
			self.updateW() 
			self.loss.append(self.lossfun())

	def forePropagation(self):
		for i in range(1,self.L):
			self.z[i] = self.w[i].dot(self.a[i-1])
			self.a[i] = self.actFun(self.z[i])
			self.gw[i] = self.a[i-1].T
			self.gz[i] = self.gradZ(self.a[i])
			self.ga[i-1] = self.w[i].T
		
		self.z[self.L] = self.w[self.L].dot(self.a[self.L-1])
		self.gw[self.L] = self.a[self.L-1].T
		self.ga[self.L-1] = self.w[self.L].T

	def backPropagation(self):
		'''
		currentd = self.gradLastZ(self.z[self.L],self.y)
		self.gz[3] = currentd
		for i in range(3,1,-1):
			self.gw[i] = currentd.dot(self.gw[i])
			currentd = self.ga[i-1].dot(currentd)
			self.ga[i-1] = currentd
			currentd = self.gz[i-1]*currentd
			self.gz[i-1] = currentd

		self.gw[1] = currentd.dot(self.gw[1])
		'''
		self.delta[3] = -(self.y-self.z[3])
		self.gw[3] = self.delta[3].dot(self.a[2].T)*1.0/self.trainnum
		self.delta[2] = self.w[3].T.dot(self.delta[3])*self.gz[2]
		self.gw[2] = self.delta[2].dot(self.a[1].T)*1.0/self.trainnum
		self.delta[1] = self.w[2].T.dot(self.delta[2])*self.gz[1]
		self.gw[1] = self.delta[1].dot(self.a[0].T)*1.0/self.trainnum

	def updateW(self):
		for i in range(1,self.L+1):
			self.w[i] = self.w[i] - self.stepsize*(self.gw[i]+0.1*self.w[i])
			#print self.w[i]

	def actFun(self,z):
		xn = np.copy(z)
		xn[xn < 0] = 0
		return xn

	def gradZ(self,a):
		xn = np.copy(a)
		xn[xn>0] = 1
		xn[xn<=0] = 0
		return xn
	
	def gradLastZ(self,z,y):
		xn = np.zeros(z.shape)
		idx = np.where((y==0)&(z>0))
		xn[idx] = 1.0
		idx = np.where((y==0)&(z<=0))
		xn[idx] = 0
		idx = np.where((y==1)&(z<1))
		xn[idx] = -1.0
		return xn

	def predict(self):
		a0, labels, y = self.loadmnist("training")
		result = self.w[3].dot(self.actFun(self.w[2].dot(self.actFun(self.w[1].dot(a0)))))
		result_labels = self.getY(result)
		count = 0
		for i in range(len(labels)):
			if(result_labels[i]==labels[i]):
				count += 1
		print count*1.0/self.testnum
	
	def getY(self,z3):
		m,n = z3.shape
		y = zeros((n,1),dtype=int)
		for i in range(n):
			y[i] = np.argmax(z3[:,i])
		return y

	def lossfun(self):
		tmp = zeros((10,self.trainnum))
		z = self.w[3].dot(self.actFun(self.w[2].dot(self.actFun(self.w[1].dot(self.a[0])))))
		x = np.zeros(z.shape)
		#print z
		tmp[self.y==1] = np.maximum(1-z,x)[self.y==1]
		tmp[self.y==0] = np.maximum(z,x)[self.y==0]
		return sum(sum(tmp))

	def lossValue(self,z):
		tmp = zeros((10,self.trainnum))
		x = np.zeros(z.shape)
		tmp[self.y==1] = np.maximum(1-z,x)[self.y==1]
		tmp[self.y==0] = np.maximum(z,x)[self.y==0]
		return sum(sum(tmp))

	def gradientCheck(self):
		epsilon = 0.00001
		#print "gw1:", self.gw[1]
		m,n = self.gw[1].shape
		check = np.zeros(self.gw[1].shape)
		for i in range(m):
			for j in range(n):
				tmp = np.copy(self.w[1])
				tmp[i,j] += epsilon
				new_z = self.w[3].dot(self.actFun(self.w[2].dot(self.actFun(tmp.dot(self.a[0])))))
				old_z = self.z[3]
				check[i,j] = (self.lossValue(new_z)-self.lossValue(old_z))/epsilon 
		#print "check:", check
		print "here",sum(sum(check-self.gw[1]))
			

neurons = [784,200,250,10]
trainnum = 10
testnum = 10
itnum = 100 
stepsize = 0.01
nn = NeuralNetwork(3,neurons,trainnum,testnum,itnum,stepsize,0.01)
nn.train()
nn.predict()

plt.plot(nn.loss)
plt.show()

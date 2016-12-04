import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------
class NeuralNetwork():
	
	def __init__(self,layers, neurons, trainnum, testnum, itnum, beta, gamma, epsilon):
		self.L = layers
		self.neurons = neurons
		self.trainnum = trainnum
		self.testnum = testnum
		self.itnum = itnum
		self.beta = beta
		self.gamma = gamma
		self.epsilon = epsilon
		self.mylambda = np.zeros((10,trainnum),dtype=np.float64)
		self.actType = "ReLu"
		self.aEnergy = list()
		self.zEnergy = list()
		self.lossEnergy = list()
		self.lambEnergy = list()
		
		self.w = list();
		self.w.append(0);
		for i in range(1,layers):
			self.w.append(epsilon*np.random.randn(neurons[i],neurons[i-1]))

		a0, self.labels, self.y = self.loadmnist()
		self.a = [a0]
		self.z = [0]
		self.aEnergy.append(0)
		self.zEnergy.append(0)
		for i in range(1,layers-1):
			self.z.append(self.w[i].dot(self.a[i-1]))
			self.a.append(self.actFun(self.z[i]))
			self.aEnergy.append(list())
			self.zEnergy.append(list())

		self.z.append(self.w[layers-1].dot(self.a[layers-2]))
		self.zEnergy.append(list())
		

#--------------------------------------------------------------------------

	def loadmnist(self, dataset="training", digits=np.arange(10), path="./"):
		if dataset == "training":
			fname_img = os.path.join(path, 'train-images-idx3-ubyte')
			fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
			N = self.trainnum
		elif dataset == "testing":
			fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
			fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
			N = self.testnum
		else:
			raise ValueError("dataset wrong")
	
		flbl = open(fname_lbl,'rb')
		magic_nr, size = struct.unpack(">II", flbl.read(8))
		lbl = pyarray("b",flbl.read())
		flbl.close()
		
		fimg = open(fname_img, 'rb')
		magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = pyarray("B", fimg.read())
		fimg.close()
		ind = [k for k in range(size) if lbl[k] in digits]
	
		images = np.zeros((N,rows,cols), dtype=uint8)
		labels = np.zeros((N,1),dtype=int8)
		for i  in range(N):
			images[i] = array(img[ind[i] * rows * cols: (ind[i]+1) * rows *cols]).reshape((rows,cols))
			labels[i] = lbl[ind[i]]

		re_img = np.zeros((784,N), dtype=np.float64)
		for i in range(N):
			re_img[:,i] = images[i,:,:].reshape(784)
		
		re_img -= np.mean(re_img,axis=0)
		re_img /= np.std(re_img,axis=0)

		y = np.zeros((10,N), dtype=uint8)
		for i in range(len(labels)):
			y[labels[i],i] = 1
		return re_img, labels, y

#--------------------------------------------------------------------------------------

	def trainByADMM(self):
		idmatrixes = list()
		idmatrixes.append(0)
		for i in range(1,self.L):
			idmatrixes.append(np.identity(self.w[i].T.shape[0]))
		idmatrixes.append(np.identity(self.trainnum))
		for k in range(self.itnum):
			for i in range(1,self.L-1):
				self.w[i] = self.updateW(i,idmatrixes)
				self.a[i] = self.updateA(i,idmatrixes)
				self.z[i] = self.updateZ(i)

			self.w[self.L-1] = self.updateW(self.L-1,idmatrixes)
			self.z[self.L-1] = self.updateLastZ()
			self.mylambda = self.mylambda + self.beta*(self.z[self.L-1] - self.w[self.L-1].dot(self.a[self.L-2]))
			#self.beta *= 1.05
			#self.gamma *= 1.05
			self.calEnergy()

		print self.z[self.L-1]
		print self.y

#---------------------------------------------------------------------------------------

	def updateW(self, i, idmatrixes):
		aT = self.a[i-1].T
		r = np.linalg.lstsq(self.a[i-1].dot(aT),idmatrixes[i])
		pinva = aT.dot(r[0])
		return self.z[i].dot(pinva)

#----------------------------------------------------------------------------------------

	def updateA(self, i, idmatrixes):
		wT = self.w[i+1].T
		tmp = self.beta * (wT.dot(self.w[i+1]) + self.gamma*idmatrixes[i+1])
		tmp = np.linalg.lstsq(tmp, idmatrixes[i+1])
		return tmp[0].dot(self.beta *wT.dot(self.z[i+1]) + self.gamma*self.actFun(self.z[i])) 

#-----------------------------------------------------------------------------------------

	def updateZ(self, i):
		aw = self.w[i].dot(self.a[i-1])
		
		z_s = np.copy(aw)
		z_s[z_s>0] = 0
		l_s = self.regularElementWiseCost(self.beta, self.gamma, self.a[i], aw, z_s)

		z_b = (self.gamma * self.a[i] + self.beta * aw) / (self.beta + self.gamma)
		z_b[z_b<0] = 0
		l_b = self.regularElementWiseCost(self.beta, self.gamma, self.a[i], aw, z_b)
		
		z_s[l_s > l_b] = z_b [l_s> l_b]

		return z_s

#------------------------------------------------------------------------------------------

	def updateLastZ(self):
		'''
		waL = self.w[self.L-1].dot(self.a[self.L-2])
		zL = np.copy(waL)
		
		tau = 0.1
		for i in range(20):
             #zL = np.zeros(waL.shape)
             # calculate probabilities
			zExp = np.exp(zL)
			zProb = 1.0 * zExp / np.sum(zExp, axis=0, keepdims=True)

		 # calculate gradient of z
			dLdz = zProb - self.y
			v = zL - tau * (dLdz)

		 # update
			zL = (2 * tau * self.beta * waL - tau * self.mylambda + v) / (1 + 2 * tau * self.beta)
		return zL

		'''
		z = self.z[self.L-1]
		for j in range(20):
			p = np.exp(z)
			sum_p = np.sum(p,axis=0)
			p = p/sum_p
			a = np.copy(p)
			p[self.y==1] = p[self.y==1] - 1
			f = p + self.mylambda + 2*self.beta*(z-self.w[self.L-1].dot(self.a[self.L-2]))
			dp = np.zeros((self.trainnum,10,10),dtype = np.float64)
			for i in range(self.trainnum):
				dp[i,:,:] = -a[:,i].reshape(10,1).dot(a[:,i].reshape(1,10))
				diag = (1-a[:,i])*(a[:,i])
				np.fill_diagonal(dp[i,:,:],diag)

			temp = np.sum(dp,axis=0)
			temp /= self.trainnum

			df = temp + 2*self.beta*np.identity(10) 

			z = z - np.linalg.inv(df).dot(f)
				

		'''
		p = np.exp(self.z[self.L-1])
		sum_p = np.sum(p,axis=0)
		p = p/sum_p
		p[self.y==1] = p[self.y==1] - 1
		p /= self.trainnum
		z = self.z[self.L-1]
		f = p+self.mylambda+2*self.beta*(z-self.w[self.L-1].dot(self.a[self.L-2]))
		df = p+2*self.beta*z
		for i in range(100):
			z = z - f/df
			p = np.exp(z)
			sum_p = np.sum(p,axis=0)
			p = p/sum_p
			p[self.y==1] = p[self.y==1] - 1
			p /= self.trainnum
			f = p+self.mylambda+2*self.beta*(z-self.w[self.L-1].dot(self.a[self.L-2]))
			df = p+2*self.beta*z
		'''

		return z

#--------------------------------------------------------------------------------------------

	def calEnergy(self):
		p = np.exp(self.w[self.L-1].dot(self.actFun(self.w[self.L-2].dot(self.a[0]))))
		sum_p = np.sum(p,axis=0)
		p = p/sum_p
		p = -np.log(p)
		p = p*self.y
		self.lossEnergy.append(np.sum(p)/self.trainnum)
		
		for i in range(1,self.L-1):
			p = self.z[i]-self.w[i].dot(self.a[i-1])
			p = np.sum(p**2,axis=0)
			p = np.sqrt(p)
			p = np.sum(p)/self.trainnum
			self.zEnergy[i].append(p)
			
			p = self.a[i]-self.actFun(self.z[i])
			p = np.sum(p**2,axis=0)
			p = np.sqrt(p)
			p = np.sum(p)/self.trainnum
			self.aEnergy[i].append(p)

		p = self.z[self.L-1]-self.w[self.L-1].dot(self.a[self.L-2])
		p = np.sum(p**2,axis=0)
		p = np.sqrt(p)
		p = np.sum(p)/self.trainnum
		self.zEnergy[self.L-1].append(p)

		self.lambEnergy.append(np.sum(self.z[self.L-1].T.dot(self.mylambda))/self.trainnum)
		

#-----------------------------------------------------------------------------------------

	def actFun(self,z):
		if(self.actType=="ReLu"):
			z[z<0] = 0
		return z

#-----------------------------------------------------------------------------------------
	
	def regularElementWiseCost(self,beta,gamma,a,aw,z):
		return gamma*(a-self.actFun(z))**2 + beta*(z-aw)**2

#----------------------------------------------------------------------------------------

	def predict(self):
		a0,labels,y = self.loadmnist("testing")
		tmp = a0
		for i in range(1,self.L-1):
			tmp = self.w[i].dot(tmp)
			tmp = self.actFun(tmp)

		tmp = self.w[self.L-1].dot(tmp)
		result = self.getY(tmp)
		count = 0
		for i in range(len(labels)):
			if(result[i]==labels[i]):
				count += 1
		print count*1.0/self.testnum

#-------------------------------------------------------------------------------------------

	def getY(self,tmp):
		m,n = tmp.shape
		y = zeros((n,1),dtype=int)
		for i in range(n):
			y[i] = np.argmax(tmp[:,i])
		return y

#-------------------------------------------------------------------------------------------
			
	def getEnergy(self):
		return self.aEnergy, self.zEnergy, self.lossEnergy, self.lambEnergy

#-------------------------------------------------------------------------------------------

network_layers = 3
neurons = [784,300,10]
trainnum = 600
testnum = 100
itnum = 20
beta = 1
gamma = 10
epsilon = 0.0001
#epsilon = np.sqrt(0.1/trainnum)

nn = NeuralNetwork(network_layers, neurons, trainnum, testnum, itnum, beta, gamma, epsilon)
nn.trainByADMM()
nn.predict()
ae , ze, le, be = nn.getEnergy()

for i in range(1,network_layers-1):
	plt.figure(2*i-1)
	plt.plot(ae[i])
	plt.figure(2*i)
	plt.plot(ze[i])

plt.figure(3)
plt.plot(ze[network_layers-1])
plt.figure(4)
plt.plot(le)
plt.figure(5)
plt.plot(be)
plt.show()

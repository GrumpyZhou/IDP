import numpy as np
from scipy import linalg
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------
class NeuralNetwork():
	
	def __init__(self,layers, neurons, trainnum, testnum, itnum, beta, gamma, epsilon):
		self.regWeight = 0.001
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
		self.tlambda = list()
		self.txi = list()
		
		self.w = list();
		self.w.append(0);
		for i in range(1,layers):
			self.w.append(epsilon*np.random.randn(neurons[i],neurons[i-1]))

		a0, self.labels, self.y = self.loadmnist()
		self.a = [a0]
		self.z = [0]
		self.aEnergy.append(0)
		self.zEnergy.append(0)
		self.tlambda.append(0)
		self.txi.append(0)
		for i in range(1,layers-1):
			self.z.append(self.w[i].dot(self.a[i-1]))
			self.a.append(self.actFun(self.z[i]))
			self.aEnergy.append(list())
			self.zEnergy.append(list())
			self.tlambda.append(zeros(self.z[i].shape)+1)
			self.txi.append(zeros(self.z[i].shape)+1)

		self.z.append(self.w[layers-1].dot(self.a[layers-2]))
		self.zEnergy.append(list())
		self.tlambda.append(zeros(self.z[layers-1].shape))
		

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
				self.tlambda[i] += 2*self.beta*(self.z[i]-self.w[i].dot(self.a[i-1]))
				self.txi[i] += 2*self.gamma*(self.a[i]-self.actFun(self.z[i]))
		

			self.w[self.L-1] = self.updateW(self.L-1,idmatrixes)
			tmp = self.z[self.L-1]
			self.z[self.L-1] = self.updateLastZ()
			self.tlambda[self.L-1] += 2*self.beta*(self.z[self.L-1]-self.w[self.L-1].dot(self.a[self.L-2]))
			#self.beta *= 1.05
			#self.gamma *= 1.05
			self.calEnergy()

		print self.z[self.L-1]
		print self.y

#---------------------------------------------------------------------------------------

	def updateW(self, i, idmatrixes):
		aTr = self.a[i-1].T
		asq = self.a[i-1].dot(aTr)
		ainv = linalg.inv(asq + self.regWeight * np.identity(self.w[i].shape[1]))
		w = 1/(2*self.beta)*(2*self.beta*self.z[i].dot(aTr)+self.tlambda[i].dot(aTr)).dot(ainv)
		return w 

#----------------------------------------------------------------------------------------

	def updateA(self, i, idmatrixes):
		tmp = 2*self.beta*self.w[i+1].T.dot(self.w[i+1])+2*self.gamma*np.identity(self.w[i+1].shape[1])
		tmp = np.linalg.inv(tmp)
		tmp = tmp.dot(self.w[i+1].T.dot(self.tlambda[i+1])-self.txi[i]+2*self.beta*self.w[i+1].T.dot(self.z[i+1])+2*self.gamma*self.actFun(self.z[i]))
		return tmp

#-----------------------------------------------------------------------------------------

	def updateZ(self, i):
		z = self.z[i]
		stepsize = 0.0001
		for j in range(10):
			total = np.zeros((self.a[i].shape[0],self.a[i].shape[0],self.a[i].shape[1]))
			reluresult = self.actFun(self.z[i])
			row,col = np.where(reluresult>0)
			for idx,val in enumerate(row):
				trow = val
				tcol = col[idx]
				total[trow][trow][tcol] = 1
			total = np.sum(total,axis=2)/self.a[i].shape[1]
			grad = self.tlambda[i]-total.dot(self.txi[i])+2*self.beta*(self.z[i]-self.w[i].dot(self.a[i-1]))-2*self.gamma*total.dot(self.a[i]-reluresult)
			z = z-stepsize*grad
		return z

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
		i = self.L-1
		waL = self.w[i].dot(self.a[i-1])
		zL = self.w[i].dot(self.a[i-1])
		N = zL.shape[1]
		for ite in range(25):
			zExp = np.exp(zL)
			zProb = 1.0 * zExp / np.sum(zExp,axis=0, keepdims=True)
			dLdz = (zProb - self.y) + 2*self.beta*(zL-waL) + self.tlambda[i];
			diag = zProb*(1-zProb)
			H = np.zeros((zL.shape[0],zL.shape[0]))
			for j in range(0,zL.shape[1]):
				pi = zProb[:,j].reshape(zL.shape[0],1)
				dLdzi = -1*pi.dot(pi.T)
				np.fill_diagonal(dLdzi,diag[:,i])
				H += dLdzi
			H = H/N+2*beta*np.identity(zL.shape[0])
			zL = zL - 0.01*np.linalg.inv(H).dot(dLdz)
		return zL
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
trainnum = 100
testnum = 100
itnum = 3
beta = 0.01
gamma = 0.1
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

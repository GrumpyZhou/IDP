import os, struct
from array import array as pyarray
import numpy as np
from pylab import *
from numpy import *

IMG_NUM = 100
FIRST_LAYER = 100
SEC_LAYER = 100
SIZE = 784

def load_mnist(dataset="training", digits=np.arange(10), path="."):
	if dataset == "training":
		fname_img = os.path.join(path, 'train-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
	elif dataset == "testing":
		fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
		fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
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

	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)
	N = IMG_NUM 

	images = zeros((N, rows, cols), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i in range(N):
		images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
		labels[i] = lbl[ind[i]]

	re_img = zeros((rows*cols,N),dtype=uint8)
	for i in range(N):	
		re_img[:,i] = images[i,:,:].reshape(rows*cols)


	y = zeros((10,N),dtype=uint8)
	print(labels)
	for i in range(N):
		y[labels[i],i] = 1

	return re_img, y

def actFun(z):
	tmp = z>=1
	z[tmp] = 1
	tmp = z<=0
	z[tmp] = 0
	return z

def getY(a3):
	m,n = a3.shape
	y = zeros((n,1),dtype=int)
	for i in range(n):
		y[i] = np.argmax(a3[:,i])
	return y

def updateW(z,a):
	return z.dot(np.linalg.pinv(a))

def updateA(beta,w,gama,z,z_p):
	m,n = w.shape
	I = np.identity(n)
	tmp = np.linalg.inv(beta*w.T.dot(w)+gama*I)
	a = tmp.dot(beta*(w.T.dot(z))+gama*actFun(z_p))
	return a

def updateZ(gama,a_c,beta,w,a_p):
	result = w.dot(a_p)
	m,n = result.shape
	z = zeros((m,n),dtype=np.float64)
	can_z = zeros((3,1),dtype=np.float64)
	for i in range(m):
		for j in range(n):
			if result[i,j]>=1:
				can_z[0] = result[i,j]
				can_z[1] = (gama*a_c[i,j]+beta*result[i,j])/(gama+beta)
				can_z[2] = 0
			elif result[i,j]<=0:
				can_z[0] = 1
				can_z[1] = (gama*a_c[i,j]+beta*result[i,j])/(gama+beta)
				can_z[2] = result[i,j]
			else:
				can_z[0] = 1
				can_z[1] = result[i,j]
				can_z[2] = 0

			if (can_z[1]>=1) or (can_z[1]<=0):
				can_z[1] = can_z[0]

			energy = zeros((3,1),dtype=np.float64)
			for k in range(3):
				if can_z[k]<=0:
					tmp = 0
				elif can_z[k]>=1:
					tmp = 1
				else:
					tmp = can_z[k]
				energy[k] = beta*pow((can_z[k]-result[i,j]),2) + gama *pow((a_c[i,j]-tmp),2)

			z[i,j] = can_z[np.argmin(energy)]

	return z

def calLastEn(z,beta,const):
	if z>0:
		tmp = z
	else:
		tmp = 0
	return tmp+beta*pow(z-const,2)

def updateLastZ(y,beta,w,a):
	m,n = y.shape
	result = w.dot(a) 
	z = zeros((m,n),dtype=np.float64)
	for i in range(m):
		for j in range(n):
			if y[i,j]==0:
				if result[i,j]<=0:
					z[i,j] = result[i,j]
				else:
					can_z = np.array([0.,0.]) 
					can_z[0] = -1/2/beta + result[i,j]
					can_z[1] = 0
					if calLastEn(can_z[0],beta,result[i,j])<calLastEn(can_z[1],beta,result[i,j]):
						z[i,j] = can_z[0]
					else:
						z[i,j] = can_z[1]
			else:
				if result[i,j]>=1:
					z[i,j] = result[i,j]
				else:
					can_z = np.array([0.,0.])
					can_z[0] = 1/2/beta + result[i,j]
					can_z[1] = 0
					if calLastEn(1-can_z[0],beta,reulst[i,j])<calLastEn(1-can_z[1],beta,result[i,j]):
						z[i,j] = can_z[0]
					else:
						z[i,j] = can_z[1]
	
	return z

def lossFun(z,y):
	m,n = z.shape
	result = zeros((m,n),dtype=np.float64)
	for i in range(m):
		for j in range(n):
			if y[i,j] == 0:
				result[i,j] = max(z[i,j],0)
			else:
				result[i,j] = max(1-z[i,j],0)
				
	return sum(sum(result))

def calEnergy(w,a,z,y,beta,gama):
	energy = 0
	energy += lossFun(z[3],y)
	tmp = sum(sum((z[3]-w[3].dot(a[2]))**2))
	energy += beta*tmp
	for i in range(1,3):
		tmp = sum(sum((a[i]-actFun(z[i]))**2))
		energy += gama*tmp
		tmp = sum(sum((z[i]-w[i].dot(a[i-1]))**2))
		energy += beta*tmp
	return energy

a0, y = load_mnist()
w = list()
z = list()
a = list()
gama = 10
beta = 1
a.append(a0)
w.append(0)
z.append(0)
w.append(np.ones((FIRST_LAYER,SIZE),dtype=np.float64))
w.append(np.ones((SEC_LAYER,FIRST_LAYER),dtype=np.float64))
w.append(np.ones((10,SEC_LAYER),dtype=np.float64))
z.append(w[1].dot(a[0]))
a.append(actFun(z[1]))
z.append(w[2].dot(a[1]))
a.append(actFun(z[2]))
z.append(w[3].dot(a[2]))

print("a: ", len(a),a[0].shape,a[1].shape,a[2].shape)
print("w: ", len(w),w[1].shape,w[2].shape,w[3].shape)
print("z: ", len(z),z[1].shape,z[2].shape,z[3].shape)

for i in range(50):
	for l in range(1,3):
		w[l] = updateW(z[l],a[l-1])
		a[l] = updateA(beta,w[l+1],gama,z[l+1],z[l])
		z[l] = updateZ(gama,a[l],beta,w[l],a[l-1])

	w[3] = updateW(z[3],a[2])
	z[3] = updateLastZ(y,beta,w[3],a[2])
	y_train = getY(z[3])
	beta = beta*1.05
	gama = gama*1.05
	
	print(calEnergy(w,a,z,y,beta,gama))

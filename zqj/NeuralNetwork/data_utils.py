import os
import struct
import numpy as np
import datetime


def saveWeight(w, path='test/saved_weight'):
    today = datetime.datetime.now().strftime ("%Y%m%d")
    fpath = os.path.join(path,'weight_%s.npy' % today)
    print fpath
    np.save(fpath, w)
    print 'Weight saved in %s'%fpath
    
def loadWeight(self, path='test/saved_weight/weight.npy'):
    print 'Load weight from %s'%path
    w = np.load(path)
    return w.tolist()

def getMnistData(path):
    
    Xtr,Ytr = read("training",path)
    Xte,Yte = read("testing",path)
    Xtr = Xtr.astype(np.float64)
    Ytr = Ytr.astype(np.int64)
    Xte = Xte.astype(np.float64)
    Yte = Yte.astype(np.int64)

    return Xtr,Ytr,Xte,Yte

def read(dataset = "training", path="."):

    if dataset is "training":
        fnameImg = os.path.join(path, 'train-images-idx3-ubyte')
        fnameLbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fnameImg = os.path.join(path, 't10k-images-idx3-ubyte')
        fnameLbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fnameLbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fnameImg, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    return img,lbl


def getMnistDataSets(path, valSize = 0, reshape=True, transpose=True):
    
    Xtr,Ytr = read("training",path)
    Xte,Yte = read("testing",path)
    Xtr = Xtr.astype(np.float64)
    Ytr = Ytr.astype(np.int64)
    Xte = Xte.astype(np.float64)
    Yte = Yte.astype(np.int64)
    
    train = DataSet(Xtr[valSize:], Ytr[valSize:], reshape=reshape, transpose=transpose)
    test = DataSet(Xte, Yte, reshape=reshape, transpose=transpose)
    if valSize != 0:
        validation = DataSet(Xtr[:valSize], Ytr[:valSize], reshape=reshape, transpose=transpose)
        return {"train": train, "validation": validation, "test": test}
    return {"train": train, "test": test}
    
class DataSet():

    def __init__(self, images, labels, reshape=True, transpose=True):
        self.imgNum = images.shape[0]
        
        if reshape:
            images = np.reshape(images, (images.shape[0], -1))
        if transpose:
            images = images.T
 
        self.images = images
        self.labels = labels
        self.round = 0
        self.index= 0
        
    @property
    def images(self):
        return self.images
    @property
    def labels(self):
        return self.labels
    @property
    def imgNum(self):
        return self.imgNum
    @property
    def round(self):
        return self.round

    def nextBatch(self, batchSize):
        """Return the next 'batchSize' entries from this data set."""
        start = self.index
        self.index += batchSize
        if self.index > self.imgNum:
            # One round completed
            self.round += 1
            # Shuffle the data
            perm = np.arange(self.imgNum)
            np.random.shuffle(perm)
            self.images = self.images[:,perm]
            self.labels = self.labels[perm]
            # Start next round
            print 'Start round #',self.round
            start = 0
            self.index = batchSize
            assert batchSize <= self.imgNum
        end = self.index 
        print start, end
        return self.images[:,range(start,end)], self.labels[start:end]

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()



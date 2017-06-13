import scipy.io as sio
import csv
from sklearn.model_selection import KFold
import pickle

def loadMatSet(matfile):
	'''
	-- matFile: the matlab file to be loaded
	'''

	matContents = sio.loadmat('octave_a.mat')
	return matContents
	
def makeSetFromRaw(csvfile,changeableInd,indirectlyInd,unchangeableInd,costs,directions,directionDependsInd,savefile=None):
	'''
	'''
	dataset = {}

	#Load the data
	dat = pd.read_csv(csvfile, delimiter=",",header=None,low_memory=False)
	dat = np.array(dat)
	nd,pd = dat.shape

	#Get header -- assumed to be the first row
	dataset['header'] = xDat[0,:]

	#Define instance data and label
	xDat = dat[1:nd,0:pd-1]
	yDat = dat[1:nd,pd]
	
	#Create dSet1 and dSet2
	kfInit = KFold(n_splits=2,shuffle=True)
	a,b = kfInit(xDat)
	dataset['dSet1'] = xDat[a[0],:]
	dataset['dSet2'] = xDat[a[1],:]
	dataset['dSet1Label'] = yDat[a[0]]
	dataset['dSet2Label'] = yDat[a[1]]
	del a,b,xDat,yDat

	#Create d1KFoldInd
	d1KFoldInd = []
	nx,px = dataset['dSet1'].shape
	kfInit = KFold(n_splits=10,shuffle=True)
	for ind in kfInit(dataset['dSet1']):
		d1KFoldInd.append(ind[1])
	dataset['d1KFoldInd'] = d1KFoldInd
	del d1KFoldInd

	#Create d2KFoldInd
	d2KFoldInd = []
        nx,px = dataset['dSet2'].shape
        kfInit = KFold(n_splits=10,shuffle=True)
        for ind in kfInit(dataset['dSet2']):
                d2KFoldInd.append(ind[1])
        dataset['d2KFoldInd'] = d2KFoldInd
	del d2KFoldInd

	#Add user-specified
	dataset['changeableIndex'] = changeableInd
	dataset['indirectlyIndex'] =  indirectlyInd
	dataset['unchangeableIndex'] =  unchangeableInd
	dataset[''] =  costs
	dataset[''] =  directions
	dataset[''] =  directionDependsInd


	#Save if specified
	if savefile != None:
        with open(savefile, 'w') as f:
        	pickle.dump(dataset, f)
	return dataset

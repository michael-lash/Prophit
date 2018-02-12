from load_mat_set import loadMatSet, makeSetFromRaw
import pickle

class Dataset:

	def __init__(self,loadFile,changeableInd=None,indirectlyInd=None,unchangeableInd=None,costs=None,directions=None,directionDependsInd=None,saveFile=None):
	    self.fileName = loadFile
	    self.fileType = loadFile.split(".")[-1]
	    if ((self.fileType != "csv") and (self.fileType != "mat")) and (self.fileType != "pickle"):
		raise Exception('File type must be either .csv, .mat, or .pickle')
	    
	    #
	    if self.fileType == 'mat':
		self.data = loadMatSet(self.fileName)
	    elif self.fileType == 'pickle':
		with open(self.fileName) as f:
			self.data = pickle.load(f)
	    else:#Must be csv and needs processing
		self.data = makeSetFromRaw(self.fileName,changeableInd,indirectlyInd,unchangeableInd,costs,directions,directionDependsInd,savefile)

	    
	    


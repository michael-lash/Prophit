import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/anlyz_results')

from Result import IndivResult
import numpy as np
import random

def main():

	resultsF = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--noprobopt.pickle"
	datFile = "/Users/mtlash/Cause/Prophit/data/gaussBenchUpdated.pickle"
	outputPlot = '/Users/mtlash/Cause/Prophit/results/figures/benchRandInstRecs.png'
	res = IndivResult(resultsF)
	res.loadDataObj(datFile)

    	idx = res.data['dSet2Label'].reshape(res.data['dSet2'].shape[0])
	#print(np.where(idx==1)[0])
	#rID = random.choice(np.where(idx==1)[0])
	rID = 159
	print(rID)
	np.set_printoptions(precision=4)
	#np.set_printoptions(suppress=True)
	#np.savetxt(sys.stdout, res.getAvgRecs(res.data), '%5.2f')
	res.plotInstChanges(res.data,rID,outputPlot)
    	#Xpos = X[idx==1,:]
    	#Xneg = X[idx==0,:]

	print("Avg obj: "+str(np.mean(res.obj,axis=0)))
        print("Avg opt obj: "+str(np.mean(res.optObj,axis=0)))
	
	

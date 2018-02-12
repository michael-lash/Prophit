#!/usr/bin/env python2.7
import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/tenflow')
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/prepdata/')


from inverse_classif import invBackPro
from numpy import genfromtxt
import numpy as np
import pandas as pd
import pickle
from DataClass import Dataset
import tensorflow as tf
from ICClass import InverseClass
from LogIC import Logger

def main():

    budget_list = range(1,11)

    procData = False    
    dataInput = "/Users/mtlash/Cause/Prophit/data/gaussBenchUpdated.pickle"
    projDir = "/Users/mtlash/Cause/Prophit/" #This will hold the path to the directory we are working in    
    logFile = projDir+"results/CAUSEbenchIC--probopt--log.txt"
    logIt = Logger(logFile)
    #resultsFileD1 = projDir+"results/cvResultsD1.pickle" #Where the cross-validation results will be saved
    invModel = projDir+"model/Cause/D1/CausecvModelD1Bench.ckpt"
    indModel = projDir+"model/Ind/INDcvModelD1.ckpt"
    valModel = projDir+"model/Cause/D2/CausecvModelD2Bench.ckpt"

    resultsFileD2 = projDir+"results/CAUSEbenchICResults--probopt.pickle" #Where the cross-validation results will be saved
    #modelFileD2 = projDir+"model/D2/cvModelD2.ckpt"


    ### Load data ###
    if procData == True:

	my_data = Dataset(dataInput,changeableInd,indirectlyInd,unchangeableInd,costs,directions,directionDependsInd,savefile=None)

    # Otherwise, load in the processed data assumed to be the dataInput file
    else:
	my_data = Dataset(dataInput)
      
    md = my_data.data
    # Call inverse classification function
    X = md['dSet2']
    nx,px = X.shape

    xDict = {}
    probDict = {}
    probDict[0] = np.zeros((nx,len(md['changeableIndex'][0])))
    objMat = np.zeros((nx,(len(budget_list)+1)))
    optObjMat = np.zeros((nx,(len(budget_list)+1)))

    for i in range(nx):

	icVar = InverseClass(x=X[i],
                                cost = md['costChange'][0],
                                uncInd = md['unchangeableIndex'][0]-1,
                                indInd = md['indirectlyIndex'][0]-1,
                                dirInd = md['changeableIndex'][0]-1,
				dirDepInd = md['directionDependsInd'],
				dirDepCut = md['directionDependsCutoff'],
                                budget = budget_list[0],
                                directions = md['increaseCost'][0],
                                invModFile=invModel,
                                indModFile=indModel,
				gPro = md['gproMods'],
				valModFile=valModel,
				doCausal=True,
				reassessProbs=True)

	icVar.setBounds()


	for j in range(len(budget_list)):


		
		# Set budget here
		icVar.budget= budget_list[j]
		# Check to see if entry created in xDict for this budget
		if budget_list[j] not in xDict.keys():
			xDict[budget_list[j]] = np.zeros((nx,len(md['changeableIndex'][0])+len(md['indirectlyIndex'][0])))
			probDict[budget_list[j]] = np.zeros((nx,len(md['changeableIndex'][0])))


		logIt.startTime()
		icVar,obj,objAct = invBackPro(icVar)
		logIt.finishTime()
		xDict[budget_list[j]][i] = np.hstack((icVar.optIndX,icVar.optx[np.newaxis]))
		#print((icVar.probs[0].shape))
		probDict[budget_list[j]][i] = icVar.probs.ravel()
		if j == 0:
			objMat[i,j] = objAct[0]
			optObjMat[i,j] = obj[0]
			probDict[0][i] = icVar.origProbs.ravel()			

		objMat[i,j+1] = objAct[-1]
		optObjMat[i,j+1] = obj[-1]
		logIt.log("Just finished inst # %s out of %s for budget %s of %s. Single time: %s minutes. Average time: %s minutes. Total time: %s minutes."% (str(i),str(nx),str(j),str(len(budget_list)),str(logIt.dTime),str(logIt.averageTime()),str(logIt.totalTime())))
		#Need to write the method for the update s.t. butdget[j+1]'= budget[j+1]-budget[j]
		#and initx is updated to the optx, etc. for the next iteration.
		
		#Update instance for next budget to get smooth transitions between recommendations
		icVar.updateInstance()

    with open(resultsFileD2, 'w') as f:
            pickle.dump({'xIndDir':xDict, 'obj':objMat, 'budgets':budget_list, 'optObj':optObjMat, 'apsDict':probDict}, f)
    return xDict, objMat, optObjMat

#!/usr/bin/env python2.7
import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/tenflow')
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/prepdata/')


from kFoldEvalParam import kFoldEval
from numpy import genfromtxt
import numpy as np
import pandas as pd
import pickle
from DataClass import Dataset
import tensorflow as tf

def main():
    hidLayer = [[50],[100],[200],[20,10],[100,50],[10,50],[50,20],[40,30,20],[100,50,20],[30,20,10]] #Different architectures to test
    maxIters = 1000 #Max Iterations before neural net training terminates
    k=10
    batchProp = .05

    procData = False    

    dataInput = "/Users/mtlash/Cause/Prophit/data/gaussBenchUpdated.pickle"
    projDir = "/Users/mtlash/Cause/Prophit/" #This will hold the path to the directory we are working in    
    resultsFileD1 = projDir+"results/CausecvResultsD1Bench.pickle" #Where the cross-validation results will be saved
    modelFileD1 = projDir+"model/Cause/D1/CausecvModelD1Bench.ckpt"


    resultsFileD2 = projDir+"results/CausecvResultsD2Bench.pickle" #Where the cross-validation results will be saved
    modelFileD2 = projDir+"model/Cause/D2/CausecvModelD2Bench.ckpt"


    ### Load data ###
    if procData == True:

	my_data = Dataset(dataInput,changeableInd,indirectlyInd,unchangeableInd,costs,directions,directionDependsInd,savefile=None)

    # Otherwise, load in the processed data assumed to be the dataInput file
    else:
	my_data = Dataset(dataInput)
     
    gProMods = my_data.data['gproMods']
    X = my_data.data['dSet1']
    X_D = X[:,my_data.data['changeableIndex'][0]-1]
    print((X_D.shape))
    nnn,ppp = X_D.shape
    X_D_p = np.zeros((nnn,ppp))
    largest = -np.inf
    smallest = np.inf
    for i in range(nnn):
	for j in range(ppp):
		est_ij,std_ij = gProMods['gp'+str(j)].predict(X[i,my_data.data['unchangeableIndex'][0]-1],return_std=True)
		X_D_p[i,j] = (1/(np.sqrt(2*np.pi)*std_ij))*(np.exp(-1*((X_D[i,j]-est_ij)**2)/(2*(std_ij**2))))
		X_D[i,j] = X_D[i,j] * X_D_p[i,j]
		if X_D[i,j] > largest:
			largest = X_D[i,j]
		if X_D[i,j] < smallest:
			smallest = X_D[i,j]

    print("The smallest prob*val is: "+str(smallest))
    print("The largest prob*val is: "+str(largest))

    X[:,my_data.data['changeableIndex'][0]-1] = X_D



    ##### Begin cross-validation test ######
    # Do the test
    resDictD1 = kFoldEval(X,my_data.data['dSet1Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD1,saveModels=modelFileD1,
			nodes='relu')
    #resDictD2 = kFoldEval(my_data.data['trainDSet2'],my_data.data['dSet2Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD2,saveModels=modelFileD2)
    print('D1 Params: '+str(resDictD1['params']))
    print('D1 AUC: '+str(resDictD1['auc']))
    print('D1 ACC: '+str(resDictD1['acc']))
    print('D1 preds: '+str(resDictD1['preds']))
    print('D1 Opt params: '+str(resDictD1['best']))

    #tf.reset_default_graph()
    X2 = my_data.data['dSet2']
    X_D2 = X2[:,my_data.data['changeableIndex'][0]-1]
    print((X_D2.shape))
    nnn,ppp = X_D2.shape
    X_D_p2 = np.zeros((nnn,ppp))
    largest = -np.inf
    smallest = np.inf
    for i in range(nnn):
        for j in range(ppp):
                est_ij,std_ij = gProMods['gp'+str(j)].predict(X2[i,my_data.data['unchangeableIndex'][0]-1],return_std=True)
                X_D_p2[i,j] = (1/(np.sqrt(2*np.pi)*std_ij))*(np.exp(-1*((X_D2[i,j]-est_ij)**2)/(2*(std_ij**2))))
                X_D2[i,j] = X_D2[i,j] * X_D_p2[i,j]
                if X_D2[i,j] > largest:
                        largest = X_D[i,j]
                if X_D2[i,j] < smallest:
                        smallest = X_D2[i,j]

    print("The smallest d2 prob*val is: "+str(smallest))
    print("The largest d2 prob*val is: "+str(largest))

    X2[:,my_data.data['changeableIndex'][0]-1] = X_D2

    

    resDictD2 = kFoldEval(X2,my_data.data['dSet2Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD2,saveModels=modelFileD2,
			nodes='relu')
    #Train models

    print('D2 Params: '+str(resDictD2['params']))
    print('D2 AUC: '+str(resDictD2['auc']))
    print('D2 ACC: '+str(resDictD2['acc']))
    print('D2 preds: '+str(resDictD2['preds']))
    print('D2 Opt params: '+str(resDictD2['best']))


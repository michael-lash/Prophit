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

    dataInput = "/Users/mtlash/Cause/Prophit/data/normBenchDSetsAllUpdated.mat"
    projDir = "/Users/mtlash/Cause/Prophit/" #This will hold the path to the directory we are working in    
    resultsFileD1 = projDir+"results/cvResultsD1relu.pickle" #Where the cross-validation results will be saved
    modelFileD1 = projDir+"model/relu/D1/cvModelD1.ckpt"

    resultsFileD2 = projDir+"results/cvResultsD2relu.pickle" #Where the cross-validation results will be saved
    modelFileD2 = projDir+"model/relu/D2/cvModelD2.ckpt"


    ### Load data ###
    if procData == True:

	my_data = Dataset(dataInput,changeableInd,indirectlyInd,unchangeableInd,costs,directions,directionDependsInd,savefile=None)

    # Otherwise, load in the processed data assumed to be the dataInput file
    else:
	my_data = Dataset(dataInput)
      
    ##### Begin cross-validation test ######
    # Do the test
    resDictD1 = kFoldEval(my_data.data['dSet1'],my_data.data['dSet1Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD1,
			saveModels=modelFileD1,nodes='relu')
    #resDictD2 = kFoldEval(my_data.data['trainDSet2'],my_data.data['dSet2Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD2,saveModels=modelFileD2)
    print('D1 Params: '+str(resDictD1['params']))
    print('D1 AUC: '+str(resDictD1['auc']))
    print('D1 ACC: '+str(resDictD1['acc']))
    print('D1 preds: '+str(resDictD1['preds']))
    print('D1 Opt params: '+str(resDictD1['best']))

    #tf.reset_default_graph()

    resDictD2 = kFoldEval(my_data.data['dSet2'],my_data.data['dSet2Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD2,saveModels=modelFileD2,
			nodes='relu')
    #Train models

    print('D2 Params: '+str(resDictD2['params']))
    print('D2 AUC: '+str(resDictD2['auc']))
    print('D2 ACC: '+str(resDictD2['acc']))
    print('D2 preds: '+str(resDictD2['preds']))
    print('D2 Opt params: '+str(resDictD2['best']))


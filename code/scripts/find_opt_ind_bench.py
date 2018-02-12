#!/usr/bin/env python2.7
import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/tenflow')
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/prepdata/')


from kFoldInd import kFoldEval
from numpy import genfromtxt
import numpy as np
import pandas as pd
import pickle
from DataClass import Dataset
import tensorflow as tf

def main():
    hidLayer = [[10],[20],[50],[100],[200],[20,10],[100,50],[10,50],[50,20]] #Different architectures to test
    maxIters = 2500 #Max Iterations before neural net training terminates
    k=10
    batchProp = .05

    procData = False    

    dataInput = "/Users/mtlash/Cause/Prophit/data/normBenchDSetsAllUpdated.mat"
    projDir = "/Users/mtlash/Cause/Prophit/" #This will hold the path to the directory we are working in    
    resultsFileD1 = projDir+"results/INDcvResultsD1.pickle" #Where the cross-validation results will be saved
    modelFileD1 = projDir+"model/Ind/INDcvModelD1.ckpt"

    #resultsFileD2 = projDir+"results/cvResultsD2.pickle" #Where the cross-validation results will be saved
    #modelFileD2 = projDir+"model/D2/cvModelD2.ckpt"


    ### Load data ###
    if procData == True:

	my_data = Dataset(dataInput,changeableInd,indirectlyInd,unchangeableInd,costs,directions,directionDependsInd,savefile=None)

    # Otherwise, load in the processed data assumed to be the dataInput file
    else:
	my_data = Dataset(dataInput)
      
    ##### Begin cross-validation test ######
    # Do the test
    xDataA = my_data.data['dSet1']
    nx,px = np.array(xDataA).shape
    print(nx,px)

    #Indices were originally created for Matlab which begins indexing at "1"; python begins at "0"....
    indsU = my_data.data['unchangeableIndex'][0] - 1
    indsD = my_data.data['changeableIndex'][0] - 1
    indsI = my_data.data['indirectlyIndex'][0] - 1
    datU = xDataA[:,indsU]
    datD = xDataA[:,indsD]
    xData1 = np.hstack((xDataA[:,indsU],xDataA[:,indsD]))
    nx,px = np.array(xData1).shape
    yData1 = my_data.data['dSet1']
    yData1 = yData1[:,indsI]
    ny, py = yData1.shape
    resDictD1 = kFoldEval(xData1,yData1,k,hidLayer,maxIters,batchProp,saveRes=resultsFileD1,saveModels=modelFileD1)
    #resDictD2 = kFoldEval(my_data.data['trainDSet2'],my_data.data['dSet2Label'],k,hidLayer,maxIters,batchProp,saveRes=resultsFileD2,saveModels=modelFileD2)
    print('D1 Params: '+str(resDictD1['params']))
    print('D1 MSE: '+str(resDictD1['mse']))
    print('D1 MAE: '+str(resDictD1['mae']))
    print('D1 Opt params: '+str(resDictD1['best']))






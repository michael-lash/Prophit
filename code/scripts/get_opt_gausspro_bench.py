#!/usr/bin/env python2.7
import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/gausspro')
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/prepdata/')


from gaussProMod import learn_gpro
from numpy import genfromtxt
import numpy as np
import pandas as pd
import pickle
from DataClass import Dataset
import tensorflow as tf

def main():
    k=10

    procData = False    

    dataInput = "/Users/mtlash/Cause/Prophit/data/normBenchDSetsAllUpdated.mat"
    projDir = "/Users/mtlash/Cause/Prophit/" #This will hold the path to the directory we are working in    
    #resultsFileD1 = projDir+"results/cvResultsD1.pickle" #Where the cross-validation results will be saved
    #modelFileD1 = projDir+"model/D1/cvModelD1.ckpt"

    updateFile = projDir+"data/gaussBenchUpdated.pickle"



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

    #Indices were originally created for Matlab which begins indexing at "1"; python begins at "0"....
    indsU = my_data.data['unchangeableIndex'][0] - 1
    indsD = my_data.data['changeableIndex'][0] - 1
    datU = xDataA[:,indsU]
    datD = xDataA[:,indsD]
    
    modelDict,lossDict = learn_gpro(datU,datD,k)
    print(lossDict)
    #Now we need to add the model and loss to the my_data structure and save it....
    my_data.data['gproMods'] = modelDict
    my_data.data['gproModsKLoss'] = lossDict

    with open(updateFile, 'w') as f:
            pickle.dump(my_data.data, f)


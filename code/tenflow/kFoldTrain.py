from sklearn.model_selection import KFold
from sklearn import metrics
from neural_net2 import learn_model
import pickle
import numpy as np
import tensorflow as tf
import os,sys
np.set_printoptions(threshold=np.nan)
#hidLayer = [[100],[200],[500],[100,100],[200,200],[100,200],[200,100]]
#maxIters = 2000

def kFoldEval(X,y,k,hidLayer,maxIters,batchProp,saveRes=None,saveModels=None,nodes='sigmoid',logger=None):
    kf = KFold(n_splits=k)
    allAcc = []
    allAuc = []
    allPreds = []
    #print(X.shape)
    #print(y.shape)
    bestParam = hidLayer
    hlCount=-1
    bAuc = 0.0
    impInd = saveModels.rfind("/")
    #Do the k fold validation
    nx,px = X.shape
    ny,py = y.shape
    batchSize = round(nx * batchProp)
    print("The col maxes: "+str(np.amax(X,axis=0)))
    print("The col mins: "+str(np.amin(X,axis=0)))
    cX = np.clip(X,np.min(np.amin(X,axis=0)),1000000000)
    # Learn a model using the best architecture
    learn_model(cX,y,bestParam,batchSize,maxIters,saveMod=saveModels,bTerm=True)
    #tf.reset_default_graph()
    saveRes = None
    if saveRes != None:
        with open(saveRes, 'w') as f:
            pickle.dump({'auc':allAuc, 'acc':allAcc, 'best':bestParam,'params':hidLayer,'preds':allPreds}, f)
    return {'auc':allAuc, 'acc':allAcc, 'best':bestParam,'params':hidLayer,'preds':allPreds}



from sklearn.model_selection import KFold
from sklearn import metrics
from neural_net_indirect import learn_model_indirect
import pickle
import numpy as np
import tensorflow as tf
import os,sys
#hidLayer = [[100],[200],[500],[100,100],[200,200],[100,200],[200,100]]
#maxIters = 2000

def kFoldEval(X,y,k,hidLayer,maxIters,batchProp,saveRes=None,saveModels=None):
    kf = KFold(n_splits=k)
    allMSE = []
    allMAE = []
    #print(X.shape)
    #print(y.shape)
    bestParam = []
    hlCount=-1
    bMSE = np.inf
    #Do the k fold validation
    for hl in hidLayer:
	hlCount+=1
	kMSE = []
	kMAE = []
        #For each training set we will search across the
        # specified hidden layers for the best result
	for train_index, test_index in kf.split(X):
	    X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            nx,px = X_train.shape
            ny,py = y_train.shape
            batchSize = round(nx * batchProp)

            trModel,x_d,y_t,k_p = learn_model_indirect(X_train,y_train,hl,batchSize,maxIters,
							saveMod=saveModels,bTerm=True)
	    saver = tf.train.Saver()
            with tf.Session() as sess:
		saver.restore(sess,saveModels)
		preds = sess.run(trModel,feed_dict={x_d:X_test,k_p:1.0})
		
		#disc_preds = sess.run(tf.greater(preds,0.5)).astype(np.int64)
		#fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=1)
		mse_val = np.mean(np.mean((preds-y_test)**2,axis=0))
		mae_val = np.mean(np.mean(np.abs(preds-y_test),axis=0))
		#auc_val = metrics.auc(fpr, tpr)
		#acc_val = metrics.accuracy_score(y_test,disc_preds)

	    kMAE.append(mae_val)
	    kMSE.append(mse_val)
            #Compare average of averages new to the best so far

	mMSE = np.mean(np.array(kMSE))
	mMAE = np.mean(np.array(kMAE))
	allMSE.append(mMSE)
	allMAE.append(mMAE)
	np.mean(mMSE)
        if np.mean(mMSE) < bMSE:
            #Update if better
	    bAuc = np.mean(mMSE)
            bestParam = hl
	#Delete the cross-val model
	os.system("rm %s" % (saveModels+"*",))
	os.system("rm %s" % (saveModels.rsplit("/")[0]+"/checkpoint",))


    # Learn a model using the best architecture
    learn_model_indirect(X,y,bestParam,batchSize,maxIters,saveMod=saveModels,bTerm=True)
    tf.reset_default_graph()
    if saveRes != None:
        with open(saveRes, 'w') as f:
            pickle.dump({'mse':allMSE, 'mae':allMAE, 'best':bestParam,'params':hidLayer}, f)
    return {'mse':allMSE, 'mae':allMAE, 'best':bestParam,'params':hidLayer}



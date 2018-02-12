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
    bestParam = []
    hlCount=-1
    bAuc = 0.0
    impInd = saveModels.rfind("/")
    #Do the k fold validation
    for hl in hidLayer:
	hlCount+=1
	kAcc = []
	kAuc = []
	kPreds = []
	kCounter = 0
        #For each training set we will search across the
        # specified hidden layers for the best result
	for train_index, test_index in kf.split(X):
	    kCounter+=1
	    X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            nx,px = X_train.shape
            ny,py = y_train.shape
            batchSize = round(nx * batchProp)
	    if logger !=None:
		logger.startTime()
            trModel,x_d,y_t,k_p,acc_val,auc_val = learn_model(np.nan_to_num(X_train),y_train,hl,batchSize,maxIters,
							saveMod=saveModels,nodeType=nodes,bTerm=True)
	    if logger != None:
		logger.finishTime()
		logger.log("Doing cv finished hidden layer # %s out of %s for kFold= %s of %s. Single time: %s minutes. Average time: %s minutes. Total time: %s minutes."% (str(hlCount),str(len(hidLayer)),str(kCounter),str(k),str(logger.dTime),str(logger.averageTime()),str(logger.totalTime())))
	    #tf.reset_default_graph()
	    saver = tf.train.Saver()
            with tf.Session() as sess:
		saver.restore(sess,saveModels)
		print(X_test)
		preds = sess.run(trModel,feed_dict={x_d:np.nan_to_num(X_test),k_p:1.0})
		disc_preds = sess.run(tf.greater(np.nan_to_num(preds),0.5)).astype(np.int64)
		print("kCounter: "+str(kCounter))
		print("preds 1-20: "+str(preds[0:20]))
		print("disc_preds 1-20: "+str(disc_preds[0:20]))	
		fpr, tpr, thresholds = metrics.roc_curve(np.nan_to_num(y_test), np.nan_to_num(preds), pos_label=1)
		auc_val = metrics.auc(fpr, tpr)
		acc_val = metrics.accuracy_score(y_test,disc_preds)
		print("auc: "+str(auc_val))
		print("acc: "+str(acc_val))
		#tf.reset_default_graph()
		#tf.reset_default_graph()
	    sess.close()
	    kAcc.append(acc_val)
	    kAuc.append(auc_val)
	    kPreds.append(np.mean(np.nan_to_num(np.array(preds))))
            #Compare average of averages new to the best so far

	mAcc = np.mean(np.array(kAcc))
	mAuc = np.mean(np.array(kAuc))
	mPred = np.mean(np.array(kPreds))
	allAuc.append(mAuc)
	allAcc.append(mAcc)
	allPreds.append(mPred)
        if mAuc > bAuc:
            #Update if better
	    bAuc = mAuc
            bestParam = hl
	#Delete the cross-val model
	os.system("rm %s" % (saveModels+"*",))
        print(saveModels.rsplit("/")[0]+"/checkpoint")
	try:
		os.system("rm %s" % (saveModels.rsplit("/")[0]+"/checkpoint",))
	except:
		print("Trouble removing /checkpoint")

    # Learn a model using the best architecture
    learn_model(X,y,bestParam,batchSize,maxIters,saveMod=saveModels,bTerm=True)
    tf.reset_default_graph()
    if saveRes != None:
        with open(saveRes, 'w') as f:
            pickle.dump({'auc':allAuc, 'acc':allAcc, 'best':bestParam,'params':hidLayer,'preds':allPreds}, f)
    return {'auc':allAuc, 'acc':allAcc, 'best':bestParam,'params':hidLayer,'preds':allPreds}



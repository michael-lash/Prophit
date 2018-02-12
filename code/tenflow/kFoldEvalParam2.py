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
            learn_model(np.clip(X_train,np.min(np.amin(X_train,axis=0)),10000),y_train,hl,batchSize,maxIters,
							saveMod=saveModels,nodeType=nodes,bTerm=True)
	    if logger != None:
		logger.finishTime()
		logger.log("Doing cv finished hidden layer # %s out of %s for kFold= %s of %s. Single time: %s minutes. Average time: %s minutes. Total time: %s minutes."% (str(hlCount),str(len(hidLayer)),str(kCounter),str(k),str(logger.dTime),str(logger.averageTime()),str(logger.totalTime())))
	    #tf.reset_default_graph()
            tf.reset_default_graph()
	    graph1 = tf.Graph()
	    with tf.Session(graph=graph1) as sess:
                with graph1.as_default():
		#saver.restore(sess,saveModels)
			saver = tf.train.import_meta_graph(saveModels+".meta",
                                                               clear_devices=True)
                        saver.restore(sess,saveModels)
			#print("First test inst: "+str(X_test[1,:]))
			#print("Shape X_train: "+str((X_train.shape)))
			#print("Shape X_test: "+str((X_test.shape)))
			trModel = tf.get_collection('model')
			x_d = tf.get_collection("x_data")
                        if len(x_d) >= 1:
                                x_d = x_d[0]
                        k_p = tf.get_collection("keep_prob")
                        if len(k_p) >= 1:
                                k_p = k_p[0]
			lss = tf.get_collection('loss')
                        if len(lss) >= 1:
                                lss = lss[0]
                        tr_st = tf.get_collection('train_step')
                        if len(tr_st) >= 1:
                                tr_st = tr_st[0]
			preds = sess.run(trModel,feed_dict={x_d:np.clip(X_test,np.min(np.amin(X_train,axis=0)),10000),k_p:1.0})[0]
			disc_preds = sess.run(tf.greater(np.nan_to_num(preds),0.5)).astype(np.int64)
			#print("kCounter: "+str(kCounter))
			#print("preds 1-20: "+str(preds[0:20]))
			#print("disc_preds 1-20: "+str(disc_preds[0:20]))
			#print("The shape of y_test is: "+str((y_test.shape)))
			#print("The shape of preds is: "+str((preds.shape)))	
			fpr, tpr, thresholds = metrics.roc_curve(np.nan_to_num(y_test), np.nan_to_num(preds), pos_label=1)
			auc_val = metrics.auc(fpr, tpr)
			acc_val = metrics.accuracy_score(y_test,disc_preds)
			print("auc: "+str(auc_val))
			print("acc: "+str(acc_val))
			#tf.reset_default_graph()
			#tf.reset_default_graph()
	    sess.close()
	    os.system("rm %s" % (saveModels[:k]+"/*ckpt*",))
            os.system("rm %s" % (saveModels[:k]+"/checkpoint",))
	    #os.system("rm %s" % (saveModels+"*",))
            #print(saveModels.rsplit("/")[0]+"/checkpoint")
            #try:
            #    os.system("rm %s" % (saveModels.rsplit("/")[0]+"/checkpoint",))
            #except:
            #    print("Trouble removing /checkpoint")	
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
	os.system("rm %s" % (saveModels[:k]+"/*ckpt*",))
	os.system("rm %s" % (saveModels[:k]+"/checkpoint",))
        #print(saveModels.rsplit("/")[0]+"/checkpoint")
	#try:
	#	os.system("rm %s" % (saveModels.rsplit("/")[0]+"/checkpoint",))
	#except:
	#	print("Trouble removing /checkpoint")

    # Learn a model using the best architecturae
    print("Best Param: "+str(bestParam))
    print("Best param avg auc: "+str(bAuc))
    print("Best param avg acc: "+str(mAcc))

    learn_model(np.clip(X,np.min(np.amin(X,axis=0)),10000),y,bestParam,batchSize,maxIters,saveMod=saveModels,bTerm=True)
    tf.reset_default_graph()
    if saveRes != None:
        with open(saveRes, 'w') as f:
            pickle.dump({'auc':allAuc, 'acc':allAcc, 'best':bestParam,'params':hidLayer,'preds':allPreds}, f)
    return {'auc':allAuc, 'acc':allAcc, 'best':bestParam,'params':hidLayer,'preds':allPreds}



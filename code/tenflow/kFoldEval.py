from sklearn.model_selection import KFold
from neural_net import learn_model
import pickle
import numpy as np
import tensorflow as tf
#hidLayer = [[100],[200],[500],[100,100],[200,200],[100,200],[200,100]]
#maxIters = 2000

def kFoldEval(X,y,k,hidLayer,maxIters,batchProp,saveRes=None,saveModels=None):
    kf = KFold(n_splits=k)
    kPreds = []
    kAct = []
    kError = []
    kParams = []
    #print(X.shape)
    #print(y.shape)
    bestParam = []
    #Do the k fold validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nx,px = X_train.shape
        ny,py = y_train.shape
        batchSize = round(nx * batchProp)
        errorHold = np.empty((1,py))
	errorHold[:] = np.Inf
	predHold = np.empty((1,py))
	kAct.append(y_test)
        #For each training set we will search across the
        # specified hidden layers for the best result
	hlCount = -1
        for hl in hidLayer:
	    hlCount +=1
            trModel,x_d,y_t,k_p = learn_model(X_train,
                                      y_train,hl,batchSize,
                                      maxIters,saveMod=saveModels,bTerm=True)
            #init_op = tf.global_variables_initializer()
	    saver = tf.train.Saver()
            with tf.Session() as sess:
		saver.restore(sess,saveModels)
                #sess.run(tf.variables_initializer([x_d,y_t]))
                #sess.run(init_op)
                #temp_loss = sess.run(loss,feed_dict={x_d:evX, y_t:evY})
                preds = sess.run(trModel,feed_dict={x_d:X_test,k_p:1.0})
            #Difference in predicted and actual probabilities
            error = y_test - preds
            #Average predicted value for each time
            #print("Error: "+str(error))
            predAvg = np.mean(preds,axis=0)
            #print("Avg preds: "+str(predAvg))
            #Average difference
            errorMean = np.mean(error,axis=0)
	    #print("Avg error: "+str(errorMean))
            #Compare average of averages new to the best so far
            if np.mean(errorMean) < np.mean(errorHold):
                #Update if better
                errorHold = errorMean
                predHold = predAvg
                bestParam = hl
        kPreds.append(predHold)
        kError.append(errorHold)
	kParams.append(bestParam)

    actAvgProb = np.mean(y,axis=0)
    #print(len(actAvgProb))
    #Save the result
    if saveRes != None:
        with open(saveRes, 'w') as f:
            pickle.dump([kPreds,kError,kAct,actAvgProb,kParams], f)
    return {'predProb':kPreds, 'predErr':kError, 'actProb':actAvgProb,'kActProbs':kAct,'kParams':kParams}



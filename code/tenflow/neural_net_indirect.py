import tensorflow as tf
import numpy as np

def learn_model_indirect(xTrain,yTrain,hLayers,batchSize,numIters,saveMod,nodeType='sigmoid',bTerm=True):
    '''
    -- hLayers: A list containg the number of nodes in each hidden layer in order (e.g., [20,50,20] for three hideen layers,
             the first with 20 nodes, the second with 50 nodes, etc.

    '''
    tf.reset_default_graph()
    tol = .00001
    checkFreq = 100
    dropProb = 0.50 #Shown to induce largest regularization variance
    nx,px = xTrain.shape
    ny,py = yTrain.shape
    x_data = tf.placeholder(shape=[None, px], dtype=tf.float32,name='x_data')
    y_target = tf.placeholder(shape=[None, py], dtype=tf.float32,name='y_target')
    keep_prob = tf.placeholder(dtype=tf.float32)

    layerDict = createLayers(px,hLayers,py)
    if bTerm == True:
            biasDict = createBias(hLayers,py)

    #Assemble model
    if nodeType != 'sigmoid':
        for i in range(len(hLayers)):
                if i == 0:
                        if bTerm == True:
                                model = tf.nn.relu(tf.add(tf.matmul(x_data, layerDict['a'+str(i+1)]), biasDict['b'+str(i+1)]))
                        else:
                                model = tf.nn.relu(tf.add(tf.matmul(x_data, layerDict['a'+str(i+1)])))
                else:
                        if bTerm == True:
                                model = tf.nn.relu(tf.add(tf.matmul(model, layerDict['a'+str(i+1)]), biasDict['b'+str(i+1)]))
                        else:
                                model = tf.nn.relu(tf.add(tf.matmul(model, layerDict['a'+str(i+1)])))
    else:
        for i in range(len(hLayers)):
    	        if i == 0:
	    	        if bTerm == True:
                                model = tf.nn.sigmoid(tf.add(tf.matmul(x_data, layerDict['a'+str(i+1)]), biasDict['b'+str(i+1)]))
                        else:
                                model = tf.nn.sigmoid(tf.add(tf.matmul(x_data, layerDict['a'+str(i+1)])))

	        else:
		        if bTerm == True:
                                model = tf.nn.sigmoid(tf.add(tf.matmul(model, layerDict['a'+str(i+1)]), biasDict['b'+str(i+1)]))
                        else:
                                model = tf.nn.sigmoid(tf.add(tf.matmul(model, layerDict['a'+str(i+1)])))
     
    #Add dropout
    model = tf.nn.dropout(model, keep_prob)
                       
    #Create the layers for the output -- no sigmoid because of regression
    if bTerm == True:
        model = tf.add(tf.matmul(model, layerDict['a'+str(i+2)]), biasDict['b'+str(i+2)])
    else:
        model = tf.add(tf.matmul(model, layerDict['a'+str(i+2)]))
    

    #Define the cost function -- in this case, cross-entropy
    
    #original----
    #cross_entropy = tf.reduce_mean(
    #                        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=model),0)

    #new-----
    #loss = -(y_target * tf.log(model + 1e-12) + (1 - y_target) * tf.log( 1 - model + 1e-12))
    loss = tf.reduce_mean(tf.reduce_mean((model-y_target)**2,axis=1))#square the difference between act and pred, take the column wise avg, reduce the sum of these mean sq errors overall.
    #cross_entropy = tf.reduce_mean(tf.reduce_sum(loss))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    tf.add_to_collection("model", model)
    tf.add_to_collection("loss",loss)
    tf.add_to_collection("train_step",train_step)
    tf.add_to_collection("x_data",x_data)
    tf.add_to_collection("keep_prob",keep_prob)

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver()
    lossVec = []
    valLoss = [np.inf]
    diffVec = []
    diff = np.inf
    #Begin training
    acc_val=None
    auc_val=None
    with tf.Session() as sess:

            sess.run(init_op)
            i = -1
            while (i < numIters): #and ((diff < tol) or (i < 500)):
                    i+=1
                    rand_index = np.random.choice(int(nx), size=int(batchSize))
                    rand_x = xTrain[rand_index,:]
                    #rand_y = np.transpose([yTrain[rand_index,:]])
                    rand_y =yTrain[rand_index,:]
                    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y, keep_prob:dropProb})
                    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y, keep_prob:1.0})
                    lossVec.append(temp_loss)

		    if i%checkFreq == 0:
			check_ind = np.random.choice(int(nx),size=int(batchSize)) #Might include indices used to train. probably should fix
			check_x = xTrain[check_ind,:]
			check_y = yTrain[check_ind,:]
                        val_loss = sess.run(loss, feed_dict={x_data: check_x, y_target: check_y, keep_prob:1.0})
                        #if len(valLoss) != 0:
			preLoss = valLoss[-1]
			diff = np.mean(abs(val_loss - preLoss))
			#else:
			#	diff = np.inf
			diffVec.append(diff)
			valLoss.append(np.mean(val_loss))
		

	    #if xTest.any() != None and yTest.any() != None:
            #	preds = sess.run(model,feed_dict={x_data:xTest})
		#auc_val, auc_op = tf.metrics.auc(labels=yTest,predictions=preds)
        	#acc_val, acc_op = tf.metrics.accuracy(labels=yTest,predictions=preds)#correct_prediction/ntx
		#sess.run(tf.variables_initializer([auc_val,auc_op,acc_val,acc_op]))
	    #	auc_val= tf.metrics.auc(labels=yTest,predictions=preds)
            #    acc_val= tf.metrics.accuracy(labels=yTest,predictions=preds)#correct_prediction/ntx
            #    sess.run(tf.variables_initializer([auc_val,acc_val]))

	    #	print(preds)
	    #	print(yTest)
    	    #	auc_res, auc_op_res = sess.run((auc_val,auc_op),feed_dict={aa:yTest,bb:preds})
            #    acc_res, acc_op_res = sess.run((acc_val,acc_op),feed_dict={aa:yTest,bb:preds})

            save_path = saver.save(sess, saveMod)
            #print("Model saved in file: %s" % save_path)
            
    #prMod = tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y_target)
    prMod = model
	
    #sess.close()
    #print("train val loss: "+str(valLoss))
    #print("train diff vals: "+str(diffVec))
    return prMod,x_data,y_target,keep_prob


def createLayers(numFeats,hLayers,numOut):
    '''
    '''

    layerDict={}
    for i in range(len(hLayers)):
            if i == 0:
                    layerDict['a'+str(i+1)] = tf.Variable(tf.random_normal(
                        shape=[numFeats,hLayers[i]]))
            else:
                    layerDict['a'+str(i+1)] = tf.Variable(tf.random_normal(
                        shape=[hLayers[i-1],hLayers[i]]))
    layerDict['a'+str(i+2)] = tf.Variable(tf.random_normal(shape=[hLayers[i],numOut]))
    return layerDict

def createBias(hLayers,numOut):
    '''
    '''
    biasDict = {}
    for i in range(len(hLayers)):
            biasDict['b'+str(i+1)] = tf.Variable(tf.random_normal(shape=[hLayers[i]]))
    biasDict["b"+str(i+2)] = tf.Variable(tf.random_normal(shape=[numOut]))
    return biasDict


import tensorflow as tf
import numpy as np

def learn_model(xTrain,yTrain,hLayers,batchSize,numIters,saveMod,nodeType='sigmoid',bTerm=True):
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

    posRatio = np.sum(yTrain)/ny
    negRatio = 1 - posRatio

    x_data = tf.placeholder(shape=[None, px], dtype=tf.float32,name='x_data')
    y_target = tf.placeholder(shape=[None, py], dtype=tf.float32,name='y_target')
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

    layerDict = createLayers(px,hLayers,py)
    if bTerm == True:
            biasDict = createBias(hLayers,py)

    if nodeType != 'sigmoid':
    #Assemble model
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
                       
    #Create the layers for the output -- should be sigmoidal because of the probabilities

    if bTerm == True:
        model = tf.nn.sigmoid(tf.add(tf.matmul(model, layerDict['a'+str(i+2)]), biasDict['b'+str(i+2)]))
    else:
        model = tf.nn.sigmoid(tf.add(tf.matmul(model, layerDict['a'+str(i+2)])))
    

    #Define the cost function -- in this case, cross-entropy
    
    #original----
    #cross_entropy = tf.reduce_mean(
    #                        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=model),0)

    #new-----
    #weight positive loss by negative ratio and vice-versa -- if there are many negative instances,
    #positive mis-classifications will seem more important.
    loss = -(negRatio*y_target * tf.log(model + 1e-12) + posRatio*(1 - y_target) * tf.log( 1 - model + 1e-12))
    #loss = -(y_target * tf.log(model + 1e-12) + (1 - y_target) * tf.log( 1 - model + 1e-12))
    ##### NEW #####
    #weighting = tf.constant([posRatio,negRatio])
    #weight_by_label = tf.multiply(y_target,weighting)
    #loss = tf.multiply(weight_by_label
    #     , tf.nn.softmax_cross_entropy_with_logits(logits=model, labels= y_target, name="xent_raw"))

    #cross_entropy = tf.reduce_mean(loss)
    ##### END NEW #####
    #### NEW 2 ####
    #weightingP = tf.constant([posRatio])
    #weightingN = tf.constant([negRatio*2])
    #weighting = tf.add(tf.multiply(y_target,weightingN),tf.multiply(1-y_target,weightingP))
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target, logits=model, weights=weighting) 
    #loss = tf.nn.weighted_cross_entropy_with_logits(targets=y_target,logits=model,pos_weight=negRatio)
    cross_entropy = tf.reduce_mean(loss)
    ##### END NEW 2 #####
  

    #cross_entropy = tf.reduce_mean(tf.reduce_sum(loss))
	
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    make_preds = tf.greater(model,0.5)
    correct_prediction = tf.equal( tf.to_float(make_preds), y_target)
    #original: correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y_target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Add additional references to get a handle on later
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
                    sess.run(train_step, feed_dict={x_data: np.nan_to_num(rand_x), y_target: rand_y, keep_prob:dropProb})
                    temp_loss = sess.run(cross_entropy, feed_dict={x_data: rand_x, y_target: rand_y, keep_prob:1.0})
		    print("temp_loss:"+str(temp_loss))
                    lossVec.append(temp_loss)

		    if i%checkFreq == 0:
			check_ind = np.random.choice(int(nx),size=int(batchSize)) #Might include indices used to train. probably should fix
			check_x = xTrain[check_ind,:]
			check_y = yTrain[check_ind,:]
                        val_loss = sess.run(cross_entropy, feed_dict={x_data: check_x, y_target: check_y, keep_prob:1.0})
			preLoss = valLoss[-1]
			diff = np.mean(abs(val_loss - preLoss))
			diffVec.append(diff)
			valLoss.append(np.mean(val_loss))
		


            save_path = saver.save(sess, saveMod)
            
    prMod = model
	
    auc_res = None
    acc_res = None
    return prMod,x_data,y_target,keep_prob,acc_res,auc_res


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


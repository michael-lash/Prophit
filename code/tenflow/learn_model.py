import tensorflow as tf
import numpy as np

def learn_model(xTrain,yTrain,hLayers,batchSize,numIters,saveMod,bTerm=True):
	'''
    
	-- hLayers: A list containg the number of nodes in each hidden layer in order (e.g., [20,50,20] for three hideen layers,
             the first with 20 nodes, the second with 50 nodes, etc.

	'''


    	nx,px = xTrain.shape
    	ny,py = yTrain.shape
    	x_data = tf.placeholder(shape=[None, px], dtype=tf.float32)
    	y_target = tf.placeholder(shape=[None, py], dtype=tf.float32)

    	layerDict = createLayers(px,hLayers,py)
    	if bTerm == True:
        	biasDict = createBias(hLayers,py)

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
                            
    	#Create the layers for the output -- should be sigmoidal because of the probabilities
    	if bTerm == True:
        	model = tf.nn.sigmoid(tf.add(tf.matmul(model, layerDict['a'+str(i+2)]), biasDict['b'+str(i+2)]))
    	else:
        	model = tf.nn.sigmoid(tf.add(tf.matmul(model, layerDict['a'+str(i+2)])))
    

    	#Define the cost function -- in this case, cross-entropy
    	cross_entropy = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=model),0)
    	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    	correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y_target,1))
    	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    	init_op = tf.global_variables_initializer()
    	saver = tf.train.Saver()
    	lossVec = []

    	#Begin training
    	with tf.Session() as sess:

            	sess.run(init_op)
            	i = 0
            	while i < numIters:
                    	i+=1
                    	rand_index = np.random.choice(int(nx), size=int(batchSize))
                    	rand_x = xTrain[rand_index,:]
                    	#rand_y = np.transpose([yTrain[rand_index,:]])
                    	rand_y =yTrain[rand_index,:]
                    	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
                    	temp_loss = sess.run(cross_entropy, feed_dict={x_data: rand_x, y_target: rand_y})
                    	lossVec.append(temp_loss)

            	# Save the variables to disk.
            	#save_path = saver.save(sess, saveMod)
            	#print("Model saved in file: %s" % save_path)
            
    	#prMod = tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y_target)
    	prMod = model
    	#sess.close()
    	return prMod,x_data,y_target


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


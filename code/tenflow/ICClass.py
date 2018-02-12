import tensorflow as tf
import numpy as np
import math
np.set_printoptions(threshold=np.nan)


class ImportGraph():		   
	""" Import tf graph as it's own object """		    
	def __init__(self, loc):		        
		# Create local graph and use it in the session
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		with self.graph.as_default():
			# Import saved model from location 'loc' into local graph
		        saver = tf.train.import_meta_graph(loc + '.meta',
		                                               clear_devices=True)
		        saver.restore(self.sess, loc)
		        # Get activation function from saved collection
		        # You may need to change this in case you name it differently
		        self.model = tf.get_collection('model')
			if len(self.model) >= 1:
				self.model = self.model[0]
			self.loss = tf.get_collection('loss')
			if len(self.loss) >= 1:
				self.loss = self.loss[0]
			self.train_step = tf.get_collection('train_step')
			if len(self.train_step) >= 1:
				self.train_step = self.train_step[0]
			#self.xD = unicode('x_data','utf-8')
			self.xD = tf.get_collection("x_data")
			if len(self.xD) >= 1:
				self.xD = self.xD[0]
			self.keep_prob = tf.get_collection("keep_prob")
			if len(self.keep_prob) >= 1:
				self.keep_prob = self.keep_prob[0]

			
			#self.xD = tf.get_default_graph().get_tensor_by_name("x_data:0")
			#a= [n.name for n in tf.get_default_graph().as_graph_def().node]
                	#print(str(a))
			#b= [op.name for op in tf.get_default_graph().get_operations()]
    			#print(str(b))
		
	def pred(self, data):
		""" Make predictions using the imported model """
		# The 'x' corresponds to name of input placeholder
		return self.sess.run(self.model, feed_dict={self.xD:data,self.keep_prob:1.0})

	def getInvGrad(self,inst):
		#obj = tf.reduce(self.model)
		
        	invGrad = tf.gradients(self.model,self.xD)
		return self.sess.run(invGrad,feed_dict={self.xD:inst,self.keep_prob:1.0})[0][0]

	def getInvGradPrime(self,inst,dInvGrad):
		#expN = np.exp(np.divide(-np.square((inst.optx-inst.estD)),(2*np.square(inst.stdD))))
		#modInvGrad = np.multiply(dInvGrad,
		#		(-np.divide(np.multiply((inst.optx-inst.estD),expN),(np.multiply(inst.estD,(np.sqrt(2)*np.sqrt(np.pi)))))))
		#print("1: "+str(modInvGrad))
		
		#modInvGrad = np.multiply(modInvGrad,
		#			((1/(np.sqrt(2*np.pi)*inst.stdD))*(np.exp(-1*\
		#			(np.square(inst.optx-inst.estD))/(2*np.square(inst.stdD))))))

		#print("2: "+str(modInvGrad))
		#modInvGrad = np.multiply(modInvGrad,
		#				inst.optx)

		expN = np.exp(np.divide(-np.square((inst.optx-inst.estD)),(2*np.square(inst.stdD))))

		modInvGrad = np.multiply(-np.divide((inst.optx-inst.estD),np.multiply(np.sqrt(2*np.pi),np.power(inst.stdD,3))),expN)
		modInvGrad = np.multiply(modInvGrad,dInvGrad)

		#print("3: "+str(modInvGrad))
		return modInvGrad

	def getInvGradG(self,inst,dInvGrad):
		modInvGradP = self.getInvGradPrime(inst,dInvGrad)
		#print("Add to grad: "+str((np.multiply((inst.lam/np.square(inst.stdD)),np.sqrt(np.square(inst.optx-inst.estD))))))
		#print("Add to grad prime: "+str((np.multiply((inst.lam/np.square(inst.stdD)),inst.optx-inst.estD))))
		modInvGradG = modInvGradP +\
					(np.multiply((inst.lam/np.square(inst.stdD)),inst.optx-inst.estD))

		#Note: previously had: -1 * (np.multiply((inst.lam/np.square(inst.stdD)),inst.optx-inst.estD))

					#(np.multiply((inst.lam/np.square(inst.stdD)),np.sqrt(np.square(inst.optx-inst.estD))))

					#(np.linalg.norm(np.divide((inst.optx - inst.estD),np.square(inst.stdD)),2)*inst.lam)
		#print(modInvGradG)
		return modInvGradG

	def makeOptimizer(self,learn_rate):
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
	
	def doInvBP(self,inst):
		print(tf.trainable_variables())
		return self.sess.run(self.optimizer.minimize(self.model,global_step=None,var_list=[self.xD]),
					feed_dict={self.xD:inst,self.keep_prob:1.0})

	def compGrads(self):
		return self.sess.run(self.optimizer.compute_gradients(self.model,var_list=[self.xD]))
		


class InverseClass():
	''' The object that will be fed to the inverse classification function '''
	def __init__(self,x=None,cost=None,uncInd=None,indInd=None,dirInd=None,dirDepInd=None,dirDepCut=None,budget=3,
			directions=[],keep_recs=True,gradStep=10,maxIters=50,gradTol=.0001,
			invModFile=None,indModFile=None,gPro=None,valModFile=None,smoking=False,
			reassessProbs=False,doCausal=True, lam=.5):
		### Params ###
		self.x = x
		self.cost = cost
		self.uncInd = uncInd
		self.indInd = indInd
		self.dirInd = dirInd
		self.dirDepInd = dirDepInd
		self.dirDepCut = dirDepCut
		self.budget = budget
		self.directions = directions
		self.keepRecs = keep_recs
		self.origGStep = gradStep
		self.gStep = gradStep
		self.mIts = maxIters
		self.origGTol = gradTol
		self.gTol = gradTol
		self.gPro = gPro
		self.reProbs = reassessProbs
		self.doCausal = doCausal
		### Files ###		
		self.invModFile = invModFile
		self.indModFile = indModFile
		self.valModFile = valModFile
		### Inverse Classification-specific ###
		self.initx = x[dirInd]
		self.optx = x[dirInd]
		self.optIndX = x[indInd][np.newaxis]
		#if self.doCausal == True:
		#	self.tinitx = np.divide(self.initx,self.probs)
		#else:
		#	self.tinitx = self.initx 
		#self.tl=np.minimum(0,self.tinitx); #Assume normalized
                #self.tu=np.maximum(1,self.tinitx);
		#self.l = np.zeros(len(self.dirInd))
		#for j in range(len(self.dirDepInd)):
                #	if self.tinitx[self.dirDepInd[j]] <= self.dirDepCut[j]:
                #        	self.directions[self.dirDepInd[j]] = 1
                #	else:   
                #        	self.directions[self.dirDepInd[j]] = -1
		#
		#self.u = np.ones(len(self.dirInd))
                #
		#self.u[np.nonzero((directions>0).astype(int))] = self.tu[np.nonzero((directions>0).astype(int))] - self.tinitx[np.nonzero((directions>0).astype(int))]
		#self.u[np.nonzero((directions<0).astype(int))] = self.tinitx[np.nonzero((directions<0).astype(int))] - self.tl[np.nonzero((directions<0).astype(int))]
		#if smoking == True:
		#	if self.x[self.dirInd[self.dirDepInd[0]]]==0:
        	#		self.u[self.dirDepInd[0]]=0
		self.smoking = smoking

		self.doInitProbs = True
		self.lam= lam

	def setBounds(self):
                self.tl=np.minimum(0,self.initx); #Assume normalized
                self.tu=np.maximum(1,self.initx);
                self.l = np.zeros(len(self.dirInd))
                for j in range(len(self.dirDepInd)):
                       if self.initx[self.dirDepInd[j]] <= self.dirDepCut[j]:
                               self.directions[self.dirDepInd[j]] = 1
                       else:
                               self.directions[self.dirDepInd[j]] = -1
                
                self.u = np.ones(len(self.dirInd))
                
                self.u[np.nonzero((self.directions>0).astype(int))] = self.tu[np.nonzero((self.directions>0).astype(int))] - self.initx[np.nonzero((self.directions>0).astype(int))]
                self.u[np.nonzero((self.directions<0).astype(int))] = self.initx[np.nonzero((self.directions<0).astype(int))] - self.tl[np.nonzero((self.directions<0).astype(int))]
                if self.smoking == True:
                       if self.x[self.dirInd[self.dirDepInd[0]]]==0:
                               self.u[self.dirDepInd[0]]=0



	def estInitDirProbs(self):
		self.probs = []
		self.estD = []
		self.stdD = []
		#self.covD = []
		for i in range(len(self.gPro.keys())):
			est_i,std_i = self.gPro['gp'+str(i)].predict(self.x[self.uncInd].reshape(1,-1),return_std=True)
			self.estD.append(est_i[0])
			self.stdD.append(std_i[0])
			est_i = max(est_i, 0)
			self.probs.append((1/(np.sqrt(2*np.pi)*std_i))*(np.exp(-1*((self.optx[i]-est_i)**2)/(2*(std_i**2)))))
		self.probs = np.array(self.probs)
		self.estD = np.array(self.estD)
		self.stdD = np.array(self.stdD)
		return self.probs, self.estD, self.stdD

	def updateDirProbs(self):
		for i in range(len(self.probs)):
			#partA = (1/(np.sqrt(2*np.pi)*self.stdD[i]))
			#partB = np.exp(-1*((self.optx[i]-self.estD[i])**2)/(2*(self.stdD[i]**2)))
			#partC = partA * partB
			#print("Part A: "+str(partA))
			#print("Part B: "+str(partB))
			#print("Part C: "+str(partC))
			#self.probs[i] = partC
			self.probs[i] = np.nan_to_num(((1/(np.sqrt(2*np.pi)*self.stdD[i]))*(np.exp(-1*((self.optx[i]-self.estD[i])**2)/(2*(self.stdD[i]**2))))))
		return self.probs
	
	def applyProbs(self):
		self.modDirX = np.multiply(self.probs.T,self.optx[np.newaxis])
		return self.modDirX

	def updateInstance(self):
		'''
		The purpose of this method is to update the instance with the optimized values and 
		to reset those that were used during the previous optimization in preparation for
		the next.
		'''
		self.initx = self.optx
		self.x[self.dirInd] = self.optx

                self.tl=np.minimum(0,self.initx); #Assume normalized
                self.tu=np.maximum(1,self.initx);
                self.l = np.zeros(len(self.dirInd))
		self.u = np.ones(len(self.dirInd))
                self.u[np.nonzero((self.directions>0).astype(int))] = self.tu[np.nonzero((self.directions>0).astype(int))] - self.initx[np.nonzero((self.directions>0).astype(int))]
                self.u[np.nonzero((self.directions<0).astype(int))] = self.initx[np.nonzero((self.directions<0).astype(int))] - self.tl[np.nonzero((self.directions<0).astype(int))]
		self.gStep = self.origGStep
		self.gTol = self.origGTol
		self.doInitProbs=False
		return True


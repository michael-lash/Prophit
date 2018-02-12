#!/usr/bin/env python
#use ctypes for projection operator
import tensorflow as tf
from ICClass import ImportGraph, InverseClass
#from projection import euclidean_proj_l1ball
from simplex_projection import euclidean_proj_l1ball
from projection_simplex import projection_simplex_bisection
import numpy as np
import sys
import copy

def invBackPro(var):
	'''
	============ invBackPro ===========
	This function computes the gradient of a tensorflow network w.r.t. the input features X
	===================================
	NOTE: NEED TO ADD var.lambda to ICClass.py

	'''
	
	invGraph = ImportGraph(var.invModFile)
	indGraph = ImportGraph(var.indModFile)
	valGraph = ImportGraph(var.valModFile)
	#Estimate indirectly changeable features
	
	var.optIndX = indGraph.pred(np.hstack((var.x[var.uncInd],var.x[var.dirInd]))[np.newaxis])
	
	#Get probability of observing the initial value
	if (var.doInitProbs == True):
		var.estInitDirProbs()
		var.origProbs = copy.copy(var.probs)
		#Apply the probabilities to the directly changeable features directlyChangeable' = probs * directlyChangeable
	if var.doCausal == True:
		var.applyProbs()
	else:
		var.modDirX = var.optx[np.newaxis]#Added [np.newaxis]
	#Obtain an initial objective function
	#Note: Added the + var.lambda etc as an additional penalty function
	#np.clip(X,np.min(np.amin(X,axis=0)),1000000000)
	obj = [invGraph.pred(np.clip(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)),-1,1000000000))[0][0]]
	actObj = [valGraph.pred(np.clip(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)),-1,1000000000))[0][0]]
	#print(np.multiply((var.lam/np.square(var.stdD)),np.square(np.linalg.norm(var.estD - var.optx,2))))
	#print(np.square(np.linalg.norm(np.divide((var.estD - var.optx),(2*np.square(var.stdD))),2))*var.lam)
	#obj = [invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)))[0][0]+\
	#	np.square(np.linalg.norm(np.divide((var.estD - var.optx),(2*np.square(var.stdD))),2))*var.lam]	

#np.multiply((var.lam/np.square(var.stdD)),np.square(np.linalg.norm(var.estD - var.optx,2)))]
	#actObj = [valGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)))[0][0]+\
	#	np.square(np.linalg.norm(np.divide((var.estD - var.optx),(2*np.square(var.stdD))),2))*var.lam]
	
	#print(obj)
	#print(actObj)
	#print("Optimizing objective: "+str(obj[0]))
        #print("Validation objective: "+str(actObj[0]))


	#invGraph.makeOptimizer(.01)
	#invGraph.compGrads()
	#aaa = invGraph.doInvBP(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)))
	#print(aaa)
	#sys.exit()
	diff = float("inf")
	iterN=0

	np.set_printoptions(precision=3)


	with tf.Session() as sess:
		while ((iterN < var.mIts) and (diff > var.gTol)):
			curFull = np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX))
			iterN+=1
			# Get gradient -- inverse model
			fullGrad = invGraph.getInvGrad(curFull)
			dirGrad = fullGrad[var.dirInd]
			indGrad = fullGrad[var.indInd]

			# Get gradient -- indirect model
			#print(dirGrad)
			#fullIndGrad = indGraph.getInvGrad(hstack((var.x[var.uncInd],var.optx)))
			#dirIndGrad = fullIndGrad[len(var.uncInd):]

			##Check gradient to make sure the update matches the enforced
			## directions.

			dirGrad = invGraph.getInvGradPrime(var,dirGrad)
			dirGSign = np.sign(-1*dirGrad)
                        dirGradUp = ((dirGSign== var.directions).astype(int))*dirGrad #Force non-agreeing directions to 0
			## Check gradient of indirect model to make sure we are updating in the correct direction when accounting for tha
			####!!!!!:::: THIS WON"T WORK --- NEED TO ANALYZE GRADIENT OF EACH ind feature and the direction of change
			# it would make to the directly changeable (zero out dirGradUp if there is a disagreement)
			#dirGradUp = ((np.sign(dirIndGrad)== np.sign(dirGradUp)).astype(int)*dirGradUp)
			
			#Apply the update
			#print(str(1/var.gStep*dirGradUp))

			
			###Need to update the gradient for optx

			#tmpx = (var.modDirX + (var.gStep*dirGradUp))[0] -- changed
			tmpx = (var.optx + (var.gStep*dirGradUp))
			#if var.doCausal == True:
			#	origTmpX = np.divide(tmpx[np.newaxis],var.probs.T)
			#	origTmpX = np.nan_to_num(origTmpX)
			#else:
			#	origTmpX = tmpx[np.newaxis]
			#Now project using the c function....
			#zc = var.cost*abs(tmpx - var.initx)
			zc = var.cost*np.clip(abs(np.nan_to_num(tmpx - var.initx)),0,var.budget)
			#print("top zc: "+str(zc))
			#print("zc: "+str(zc))
			#print("cost times u: "+str(var.cost*var.u))
			#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc), var.budget,var.cost*var.u)
			#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc), var.budget)
			proj_zc = projection_simplex_bisection(np.nan_to_num(zc), z=var.budget, tau=0.0001, max_iter=1000)
			proj_z = np.divide(np.multiply(var.directions,proj_zc),var.cost)
			var.optx = var.initx + proj_z
			if var.reProbs == True:
				var.updateDirProbs()
			if var.doCausal==True:
				var.applyProbs()
			else:
				var.modDirX = var.optx[np.newaxis]
			

			#Estimate indirectly changeable
			
			var.optIndX = indGraph.pred(np.hstack((var.x[var.uncInd],var.optx))[np.newaxis])
			
			cObj = invGraph.pred(np.clip(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)),-1,1000000000))[0][0]
			#print("The cObj for itern = "+str(iterN)+" is: "+str(cObj))
			#cObj = invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd],var.optIndX)),var.modDirX)))
			while ((cObj > obj[-1]) and (var.gStep < 10000)):
				#var.gStep=1/2*var.gStep
				var.gStep = 2*var.gStep
				

				###Need to update the gradient for optx

	                        #tmpx = (var.modDirX + (var.gStep*dirGradUp))[0] -- changed
        	                tmpx = (var.optx + (1/var.gStep*dirGradUp))[0]
                	        #if var.doCausal == True:
                       		#       origTmpX = np.divide(tmpx[np.newaxis],var.probs.T)
                       		#       origTmpX = np.nan_to_num(origTmpX)
                        	#else:
                        	#       origTmpX = tmpx[np.newaxis]
                        	#Now project using the c function....
                        	#zc = var.cost*abs(tmpx - var.initx)
				zc = var.cost*np.clip(abs(np.nan_to_num(tmpx - var.initx)),0,var.budget)
				#print("zc: "+str(zc))
                        	#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc), var.budget,var.cost*var.u)
                        	#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc), var.budget)
				proj_zc = projection_simplex_bisection(np.nan_to_num(zc), z=var.budget, tau=0.0001, max_iter=1000)
				proj_z = np.divide(np.multiply(var.directions,proj_zc),var.cost)
                        	var.optx = var.initx + proj_z
                        	if var.reProbs == True:
                                	var.updateDirProbs()
                        	if var.doCausal==True:
                                	var.applyProbs()
                        	else:
                                	var.modDirX = var.optx[np.newaxis]


				var.optIndX = indGraph.pred(np.hstack((var.x[var.uncInd],var.optx))[np.newaxis])
                        	cObj = invGraph.pred(np.clip(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)),-1,1000000000))[0][0]
				#print("Improving the cObj loop the cObj is: "+str(cObj))
			#Update objective function
			obj.append(invGraph.pred(np.clip(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)),-1,1000000000))[0][0])
			#print("The updated obj value is: "+str(obj[-1]))			

			#Calculate the diff
			diff = (abs(obj[-1]-obj[-2]))/abs(obj[-2])
			#print("The diff is now: "+str(diff))


			#need to update stepsize variable "step"
			#var.gStep = var.gStep/1.5
			var.gStep = var.gStep * 1.5
	#actObj.append(valGraph.pred(np.hstack((np.hstack((var.x[var.uncInd],var.optIndX)),var.modDirX))))
	actObj.append(valGraph.pred(np.clip(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)),-1,1000000000))[0][0])
	#print("The final validation objective is: "+str(actObj[-1]))
	#sys.exit()
	

	return var, obj, actObj
		

		

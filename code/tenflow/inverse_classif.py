#!/usr/bin/env python
#use ctypes for projection operator
import tensorflow as tf
from ICClass import ImportGraph, InverseClass
#from projection import euclidean_proj_l1ball
#from simplex_projection import euclidean_proj_l1ball
from projection_simplex import projection_simplex_bisection
import numpy as np
import sys
import copy
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=20)
def invBackPro(var):
	'''
	============ invBackPro ===========
	This function computes the gradient of a tensorflow network w.r.t. the input features X
	===================================


	'''
	
	invGraph = ImportGraph(var.invModFile)
	indGraph = ImportGraph(var.indModFile)
	valGraph = ImportGraph(var.valModFile)
	#Estimate indirectly changeable features
	
	var.optIndX = indGraph.pred(np.hstack((np.nan_to_num(var.x[var.uncInd]),np.nan_to_num(var.x[var.dirInd])))[np.newaxis])
	#print("indX est: "+str(var.optIndX))
	#Get probability of observing the initial value
	if (var.doInitProbs == True):
		var.estInitDirProbs()
		var.origProbs = copy.copy(var.probs)
		#Apply the probabilities to the directly changeable features directlyChangeable' = probs * directlyChangeable
	if var.doCausal == True:
		var.applyProbs()
	else:
		var.modDirX = var.optx[np.newaxis]#Added [np.newaxis]
	#print((var.modDirX.shape))
	#Obtain an initial objective function
	obj = [invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)))[0][0]]
	actObj = [valGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX)))[0][0]]

	print("Optimizing objective: "+str(obj[0]))
        print("Validation objective: "+str(actObj[0]))


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

			dirGSign = np.sign(-1*dirGrad)
			dirGradUp = ((dirGSign== var.directions).astype(int))*dirGrad #Force non-agreeing directions to 0
			#print("Initial upper bounds: "+str(var.u))

			#print("dirGradUp: "+str(dirGradUp))

			#print("Probs: "+str(var.probs))
			## Check gradient of indirect model to make sure we are updating in the correct direction when accounting for tha
			####!!!!!:::: THIS WON"T WORK --- NEED TO ANALYZE GRADIENT OF EACH ind feature and the direction of change
			# it would make to the directly changeable (zero out dirGradUp if there is a disagreement)
			#dirGradUp = ((np.sign(dirIndGrad)== np.sign(dirGradUp)).astype(int)*dirGradUp)
			
			#Apply the update
			#print(str(1/var.gStep*dirGradUp))
			tmpx = (var.modDirX + (var.gStep*dirGradUp))[0]
			#print("tmpx: "+str(tmpx))
			if var.doCausal == True:
				#origTmpX = np.divide(tmpx[np.newaxis],var.probs.T)
				origTmpX = np.divide(tmpx[np.newaxis], var.probs.T, out=np.zeros_like(tmpx[np.newaxis]), where=var.probs.T!=0)
				origTmpX = np.nan_to_num(origTmpX)
			else:
				origTmpX = tmpx[np.newaxis]
			#Now project using the c function...
			#print("origTmpX: "+str(origTmpX))
			zc = var.cost*np.clip(abs(np.nan_to_num(origTmpX - var.initx)),0,var.budget)
			#print("zc: "+str(zc))
			#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc[0]), var.budget,var.cost*var.u)
			#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc[0]), var.budget)
			proj_zc = projection_simplex_bisection(np.nan_to_num(zc[0]), z=var.budget, tau=0.0001, max_iter=1000)
			#print("proj_zc: "+str(proj_zc))
			proj_z = np.divide(np.multiply(var.directions,proj_zc),var.cost)
			#print("proj_z: "+str(proj_z))
			var.optx = np.nan_to_num(var.initx + proj_z)
			#print("var.optx :"+str(var.optx))
			if var.reProbs == True:
				var.updateDirProbs()
			if var.doCausal==True:
				var.applyProbs()
			else:
				var.modDirX = var.optx[np.newaxis]
			

			#Estimate indirectly changeable
			
			var.optIndX = np.nan_to_num(indGraph.pred(np.hstack((var.x[var.uncInd],var.optx))[np.newaxis]))
			
			cObj = np.nan_to_num(invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX))))[0][0]
			#print("The cObj for itern = "+str(iterN)+" is: "+str(cObj))
			#cObj = invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd],var.optIndX)),var.modDirX)))
			while ((cObj > obj[-1]) and (var.gStep < 10000)):
				#var.gStep=1/2*var.gStep
				var.gStep = 2*var.gStep
				#Apply the update
				#print(str(var.gStep*dirGradUp))
				tmpx = (var.modDirX + (1/var.gStep*dirGradUp))[0]
				if var.doCausal == True:
                        		#origTmpX = np.divide(tmpx[np.newaxis],var.probs.T)
					origTmpX = np.divide(tmpx[np.newaxis], var.probs.T, out=np.zeros_like(tmpx[np.newaxis]),where=var.probs.T!=0)
					origTmpX = np.nan_to_num(origTmpX)
				else:
					origTmpX = tmpx[np.newaxis]
                        	#Now project using the c function....
                        	zc = var.cost*np.clip(abs(np.nan_to_num(origTmpX - var.initx)),0,var.budget)
                        	#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc[0]), var.budget,var.cost*var.u)
                        	#proj_zc = euclidean_proj_l1ball(np.nan_to_num(zc[0]), var.budget)
				proj_zc = projection_simplex_bisection(np.nan_to_num(zc[0]), z=var.budget, tau=0.0001, max_iter=1000)
				proj_z = np.divide(np.multiply(var.directions,np.nan_to_num(proj_zc)),var.cost)
                        	var.optx = np.nan_to_num(var.initx + proj_z)
				if var.reProbs == True:
                        		var.updateDirProbs()
				if var.doCausal == True:
                        		var.applyProbs()
				var.optIndX = np.nan_to_num(indGraph.pred(np.hstack((var.x[var.uncInd],var.optx))[np.newaxis]))
                        	cObj = np.nan_to_num(invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX))))[0][0]
				#print("Improving the cObj loop the cObj is: "+str(cObj))
			#Update objective function
			obj.append(np.nan_to_num(invGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX))))[0][0])
			#print("The updated obj value is: "+str(obj[-1]))			

			#Calculate the diff
			diff = (abs(obj[-1]-obj[-2]))/abs(obj[-2])
			#print("The diff is now: "+str(diff))


			#need to update stepsize variable "step"
			#var.gStep = var.gStep/1.5
			var.gStep = var.gStep * 1.5
	#actObj.append(valGraph.pred(np.hstack((np.hstack((var.x[var.uncInd],var.optIndX)),var.modDirX))))
	actObj.append(np.nan_to_num(valGraph.pred(np.hstack((np.hstack((var.x[var.uncInd][np.newaxis],var.optIndX)),var.modDirX))))[0][0])
	#print("The final validation objective is: "+str(actObj[-1]))
	#sys.exit()
	return var, obj, actObj
		

		

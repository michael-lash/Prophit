#!/usr/bin/env python
#use ctypes for projection operator


def invBackPro(model,x,cost,direction,Budget,step,maxIters,tol):
	'''
	============ invBackPro ===========
	This function computes the gradient of a tensorflow network w.r.t. the input features X
	===================================


	'''
	

	optx = x
	diff = float("inf")
	iterN=0
	while iterN < maxIters and diff > tol:
		iterN+=1
		
		#Obtain the gradient
		grad = invGradient(model,optx)
		#Ensure that the  direction of optimization matches the enforced direction
		enfDir = int((tf.divide(-grad,tf.multiply(direction,-grad))==direction))
		#Apply gradient
		optx = optx - (tf.multiply(step,tf.multiply(enfDir,grad))
		#Now project using the c function....
		
		#need to update stepsize variable "step"

	return optx
		

def invGradient(model,x):
	'''
	'''
	obj = tf.reduce(model)
	invGrad = tr.gradients(obj,x)
	return invGrad
		

from sklearn import gaussian_process as gp
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
from sklearn.model_selection import KFold
from copy import deepcopy
import numpy as np

def gaussDist(x,barX,stDev):
	a = 1/(stDev * np.sqrt(2*np.pi))
	b = np.exp( - (x - barX)**2 / (2 * stDev**2) )
	return a/b

	#return ((1/(np.sqrt(2*np.pi)*stDev))*(np.exp((-1*(x-barX)**2)/(2*(stDev**2
#)))))


def learn_gpro(X,Y,Xts,Yts,k):
	
	#myKern = 1.0 * RBF(length_scale=20.0, length_scale_bounds=(10, 1000))

	myKern = 1.0 * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e3)) \
    			+ WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e+1))

	X = np.nan_to_num(X)
	Xts = np.nan_to_num(Xts)

	YMod = deepcopy(Y)
	YTMod = deepcopy(Yts)

	yn,yp = Y.shape
	Xn,Xp = X.shape
	tsXn,tsXp = Xts.shape
	tsYn,tsYp = Yts.shape
	X_D_es = np.zeros((tsYn,tsYp))
        X_D_st = np.zeros((tsYn,tsYp))
	#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) #defining the kernel function. borrowed from the sklearn gp regression example
	kf = KFold(n_splits=k)
	lossDict = {'loss_type':'SSE'}
	modelDict = {}
	for i in range(yp):
		y = Y[:,i]
		#print((y.shape))
		#bLoss = np.inf #Set best BIC found to infinity.
		#bComp = 0 #Set best parameter equal to 0 for now.
		#kLoss = []
		#for train_index, test_index in kf.split(X):
			#gPro = gp.GaussianProcessRegressor(n_restarts_optimizer=3)
			#gPro.fit(X[train_index],y[train_index])
			#preds= gPro.predict(X[test_index])#,return_std=True, return_cov=True)
			#ss = np.sum((y[test_index] - preds)**2)
			#kLoss.append(ss)
		#lossDict[i] = np.mean(kLoss)
		#Train a model using all of the data
		#gPro = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
		gPro = gp.GaussianProcessRegressor(kernel=myKern,n_restarts_optimizer=3,alpha=0.0)
		gPro.fit(X,y)

		predsTr,devsTr =  gPro.predict(X,return_std=True)#,return_std=True, return_cov=True)
		predsTs,devsTs = gPro.predict(Xts,return_std=True)
		

		X_D_es[:,i] = predsTs
		X_D_st[:,i] = devsTs

		##Calc pred stuff
		avgTr = np.mean(predsTr)
		stdTr = np.std(predsTr)
		maxTr = max(predsTr)
		minTr = min(predsTr)

		avgTs = np.mean(predsTs)
		stdTs = np.std(predsTs)
		maxTs = max(predsTs)
		minTs = min(predsTs)
		
		#Get dist stuff
		#training
		trDist = []
		tsDist = []
		for j in range(Xn):
			ggVal = gaussDist(Y[j,i],predsTr[j],devsTr[j])
			YMod[j,i] = YMod[j,i] * ggVal
			trDist.append(ggVal)
		#Testing
		for j in range(tsXn):
			ggVal = gaussDist(Yts[j,i],predsTs[j],devsTs[j])
			YTMod[j,i] = YTMod[j,i] * ggVal
			tsDist.append(ggVal)

		## Calc dist stuff ##
		trDist = np.array(trDist)
		tsDist = np.array(tsDist)

		avgTrD = np.mean(trDist)
		stdTrD = np.std(trDist)

		avgTsD = np.mean(tsDist)
		stdTsD = np.std(tsDist)

		avgDevTr = np.mean(devsTr)
		stdDevTr = np.std(devsTr)

		avgDevTs = np.mean(devsTs)
		stdDevTs = np.std(devsTs)

		print("##### Feature i= "+str(i)+" #####")
		print("~Pred info~")
		print("Train STD: "+str(stdTr))
		print("Train AVG: "+str(avgTr))
		print("Train max: "+str(maxTr))
		print("Train min: "+str(minTr))

		print("Test STD: "+str(stdTs))
                print("Test AVG: "+str(avgTs))
                print("Test max: "+str(maxTs))
                print("Test min: "+str(minTs))
		
		print("###")
		print("~Dist dev info~")
		print("Train dev STD: "+str(stdDevTr))
                print("Train dev AVG: "+str(avgDevTr))

                print("Test dev STD: "+str(stdDevTs))
                print("Test dev AVG: "+str(avgDevTs))

		print("###")
		print("~Dist info~")
		print("Train dist STD: "+str(stdTrD))
                print("Train dist AVG: "+str(avgTrD))

		print("Test dist STD: "+str(stdTsD))
                print("Test dist AVG: "+str(avgTsD))
		print("################################")
		print("")
		#Store the model in the model dictionary
		modelDict['gp'+str(i)] = deepcopy(gPro)
	return modelDict, lossDict, YMod, YTMod, X_D_es, X_D_st





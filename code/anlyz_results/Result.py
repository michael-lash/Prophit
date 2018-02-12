import matplotlib as mpl
mpl.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import sys
from  itertools import product
import random

class Result():

	def __init__(self,resultFiles=[],dataFile=None):

		self.rFiles = resultFiles
		self.dFile = dataFile
		self.results = {}
		for fName in self.rFiles: 
                	self.results[fName] = IndivResult(fName)

		with open(self.dFile) as f:
			self.data = pickle.load(f)
		
		self.budgets = self.results[self.rFiles[0]].budgets


	def makeProbPlot(self,saveName,labels):
		'''
		Plot average probability of optimized instances by budget value
		'''
		xLab = r'Budget'#Latex
                yLab = r'Average iFEE' #Latex
                title = r'Average iFEE vs. Budget by Objective Function'
                #title = r'Average Actual vs Average Predicted Probability: No Geo Feats' #Latex
		
		r = 30
		

                lsList = ['-','--','-.',':']

                colList = ['r', 'g', 'b','c','m','k']

                lWidth = 2.0
                axFntSz = 12
                tiFntSz = 12


                stdMod = .25

                #Figure
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.axis([0,len(self.budgets), -0.04, .13])
                #ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']) +
                #                        cycler('linestyle', lsList))
		np.random.seed(r)
		colLin = np.array(list(product(colList,lsList)))
		np.random.shuffle(colLin)
                ## Setup latex stuff ##
                #plt.rc('text', usetex=True)
                plt.rc('font', family='serif')

		lineDict = {}
                self.budgets = np.insert(self.budgets,0,0)
		print("Budgets: "+str(self.budgets))
		nameCounter = -1
		for fName in self.rFiles:
			nameCounter +=1
			#np.set_printoptions(suppress=True)
        		#np.savetxt(sys.stdout, fName+": "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)), '%5.2f')
			print(fName+": Means: "+str(np.mean(self.results[fName].obj,axis=0)))
			print(fName+": Std dev: "+str(np.std(self.results[fName].obj,axis=0)))
			print(fName+": Diffs: "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)))
			lineDict[fName] = ax.plot(self.budgets, 
						np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0), 
						linewidth=lWidth,
						label=labels[nameCounter], color = colLin[nameCounter][0], 
						linestyle=colLin[nameCounter][1])
						
		
		plt.rcParams['axes.linewidth'] = 0.5

                #Set labels and title

                ax.set_xlabel(xLab,fontsize=axFntSz)
                ax.set_ylabel(yLab,fontsize=axFntSz)

                ax.set_title(title,fontsize=tiFntSz)

                ax.legend()

                fig.savefig(saveName,dpi=300)



	def makeCompLamPlot(self,saveName,lambdas,labels,budIndex):
		'''
		Note: the first file is assumed to be the f' and other files thereafter to be represent the lambdas in order
		that they appear in the list lambdas.

                Also, lambdas should not include "0" first -- it will be added in this code.
                budIndex == is the index of the budget-value that will be used to 
		'''
		xLab = r'Lambda'#Latex
                yLab = r'Average iFEE' #Latex
                title = r"Average iFEE vs. Lambda: f'-opt vs g"
                #title = r'Average Actual vs Average Predicted Probability: No Geo Feats' #Latex

                r = 30


                lsList = ['-','--','-.',':']

                colList = ['r', 'g', 'b','c','m','k']

                lWidth = 2.0
                axFntSz = 12
                tiFntSz = 12
		
		stdMod = .25

                #Figure
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.axis([0,len(lambdas), 0.0, 0.10])
		xAxLabsInt = [i for i in range(len(lambdas)+1)]
                #ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']) +
                #                        cycler('linestyle', lsList))
                np.random.seed(r)
                colLin = np.array(list(product(colList,lsList)))
                np.random.shuffle(colLin)
                ## Setup latex stuff ##
                #plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                lineDict = {}
		lambdas = np.array(lambdas)
                lambdas = np.insert(lambdas,0,0)
		self.budgets = np.insert(self.budgets,0,0)
		nn = len(lambdas)
                nameCounter = -1

		
                gPlot = np.zeros((nn,1))
		print("Bud value at bud index is: "+str(self.budgets[budIndex]))


                for fName in self.rFiles:
                        nameCounter +=1
                        #np.set_printoptions(suppress=True)
                        #np.savetxt(sys.stdout, fName+": "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)), '%5.2f')
                        #print(fName+": "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)))
			print("nameCounter: %s, fName: %s"%(str(nameCounter),fName,))
			if nameCounter == 0:
				val = (np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0))[budIndex]
				print("fPrime rep value is: "+str(val))
				fPrimePlot = np.ones((nn,1)) * val
				gPlot[0] = val	
				
			else:
				val = (np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0))[budIndex]
				gPlot[nameCounter] = val
                        	

		#maxVAL = max(max(fPrimePlot),max(gPlot))


	        lineDict['fPrime'] = ax.plot(xAxLabsInt,
                                        fPrimePlot,
                                        linewidth=lWidth,
                                        label=labels[0], color = colLin[0][0],
                                        linestyle=colLin[0][1])		


		lineDict['g'] = ax.plot(xAxLabsInt,
					gPlot,
                                        linewidth=lWidth,
                                        label=labels[1], color = colLin[1][0],
                                        linestyle=colLin[1][1])


		#ax.axis([0,len(lambdas), 0.0, maxVAL+1.0])

                plt.rcParams['axes.linewidth'] = 0.5

                #Set labels and title

                ax.set_xlabel(xLab,fontsize=axFntSz)
                ax.set_ylabel(yLab,fontsize=axFntSz)

                ax.set_title(title,fontsize=tiFntSz)

		a=ax.get_xticks().tolist()
		print("axis labels before: "+str(a))
		for i in range(len(a)):
			a[i] = str(lambdas[i])
		print("axis labels after: "+str(a))
		ax.set_xticklabels(a)

                ax.legend()

                fig.savefig(saveName,dpi=300)
		return fPrimePlot,gPlot


	def makeAPSLamPlot(self,saveName,lambdas,labels,budIndex):
                '''
                Note: the first file is assumed to be the f' and other files thereafter to be represent the lambdas in order
                that they appear in the list lambdas.

                Also, lambdas should not include "0" first -- it will be added in this code.
                budIndex == is the index of the budget-value that will be used to
                '''
                xLab = r'Lambda'#Latex
                yLab = r'Average APS' #Latex
                title = r"Average APS vs. Lambda: f'-opt vs g"
                #title = r'Average Actual vs Average Predicted Probability: No Geo Feats' #Latex

                r = 30


                lsList = ['-','--','-.',':']

                colList = ['r', 'g', 'b','c','m','k']

                lWidth = 2.0
                axFntSz = 12
                tiFntSz = 12

                stdMod = .25

                #Figure
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.axis([0,len(lambdas), 0.0, 0.90])
                xAxLabsInt = [i for i in range(len(lambdas)+1)]
                #ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']) +
                #                        cycler('linestyle', lsList))
                np.random.seed(r)
                colLin = np.array(list(product(colList,lsList)))
                np.random.shuffle(colLin)
                ## Setup latex stuff ##
                #plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                lineDict = {}
		lambdas = np.array(lambdas)
                lambdas = np.insert(lambdas,0,0)
                self.budgets = np.insert(self.budgets,0,0)
                nn = len(lambdas)
                nameCounter = -1


                gPlot = np.zeros((nn,1))
                print("Bud value at bud index is: "+str(self.budgets[budIndex]))


                for fName in self.rFiles:
                        nameCounter +=1
                        #np.set_printoptions(suppress=True)
                        #np.savetxt(sys.stdout, fName+": "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)), '%5.2f')
                        #print(fName+": "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)))
                        
			
			print("nameCounter: %s, fName: %s"%(str(nameCounter),fName,))
                        if nameCounter == 0:
				val = self.results[fName].computeAvgNZProp(self.data)[budIndex]
                                print("fPrime rep value is: "+str(val))
                                fPrimePlot = np.ones((nn,1)) * val
                                gPlot[0] = val

                        else:
                                val = self.results[fName].computeAvgNZProp(self.data)[budIndex]
                                gPlot[nameCounter] = val


                #maxVAL = max(max(fPrimePlot),max(gPlot))


                lineDict['fPrime'] = ax.plot(xAxLabsInt,
                                        fPrimePlot,
                                        linewidth=lWidth,
                                        label=labels[0], color = colLin[0][0],
                                        linestyle=colLin[0][1])


                lineDict['g'] = ax.plot(xAxLabsInt,
                                        gPlot,
                                        linewidth=lWidth,
                                        label=labels[1], color = colLin[1][0],
                                        linestyle=colLin[1][1])


                #ax.axis([0,len(lambdas), 0.0, maxVAL+1.0])

                plt.rcParams['axes.linewidth'] = 0.5

                #Set labels and title

                ax.set_xlabel(xLab,fontsize=axFntSz)
                ax.set_ylabel(yLab,fontsize=axFntSz)

                ax.set_title(title,fontsize=tiFntSz)
		a=ax.get_xticks().tolist()
                print("axis labels before: "+str(a))
                for i in range(len(a)):
                        a[i] = str(lambdas[i])
                print("axis labels after: "+str(a))
                ax.set_xticklabels(a)

                ax.legend()

                fig.savefig(saveName,dpi=300)
                return fPrimePlot,gPlot


	def makeCommonRecTable(self,budIndex=None):
		'''
		This function will output a latex table showing to most common recommendations by model.
		Note: may need to be modified for ARIC, which has many more changeable features.
		'''
		for fName in self.rFiles:
			if budIndex != None:
				print("Budget: "+str(self.budgets[budIndex]))
				self.results[fName].getTopRecs(self.data,budIndex)
			else:
				budgetCount = -1
				for i in self.budgets:
					budgetCount +=1
					print("Budget: "+str(i))
					self.results[fName].getTopRecs(self.data,budgetCount)


	def makeAvgAPSPlot(self,saveName,labels):
                '''
                Plot average probability of optimized instances by budget value
                '''
                xLab = r'Budget'#Latex
                yLab = r'Average APS' #Latex
                title = r'Average APS vs. Budget by Objective Function'
                	#title = r'Average Actual vs Average Predicted Probability: No Geo Feats' #Latex

                r = 30


                lsList = ['-','--','-.',':']

                colList = ['r', 'g', 'b','c','m','k']

                lWidth = 2.0
                axFntSz = 12
                tiFntSz = 12


                stdMod = .25

                #Figure
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.axis([0,len(self.budgets), 0.0, 1.3])
                #ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']) +
                #                        cycler('linestyle', lsList))
                np.random.seed(r)
                colLin = np.array(list(product(colList,lsList)))
                np.random.shuffle(colLin)
                ## Setup latex stuff ##
                #plt.rc('text', usetex=True)
                plt.rc('font', family='serif')

                lineDict = {}
                self.budgets = np.insert(self.budgets,0,0)
                print("Budgets: "+str(self.budgets))
                nameCounter = -1
			
		for fName in self.rFiles:
                        nameCounter +=1
			self.results[fName].computeAvgNZProp(self.data)
			#self.results[fName].computeAvgEstDev(self.data)
                        #np.set_printoptions(suppress=True)
                        #np.savetxt(sys.stdout, fName+": "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)), '%5.2f')
                        print(fName+": Means: "+str(self.results[fName].avgNZProp))
                        #print(fName+": Std dev: "+str(np.std(self.results[fName].obj,axis=0)))
                        #print(fName+": Diffs: "+str(np.mean(self.results[fName].obj,axis=0)[0] - np.mean(self.results[fName].obj,axis=0)))
                        lineDict[fName] = ax.plot(self.budgets,
                                                self.results[fName].avgNZProp,
                                                linewidth=lWidth,
                                                label=labels[nameCounter], color = colLin[nameCounter][0],
                                                linestyle=colLin[nameCounter][1])


                plt.rcParams['axes.linewidth'] = 0.5

                #Set labels and title

                ax.set_xlabel(xLab,fontsize=axFntSz)
                ax.set_ylabel(yLab,fontsize=axFntSz)

                ax.set_title(title,fontsize=tiFntSz)

                ax.legend()

                fig.savefig(saveName,dpi=300)

		
class IndivResult():
	
	def __init__(self,resultFile):
		self.rFile = resultFile
		with open(self.rFile) as f:
                	self.results = pickle.load(f)

		self.xVals = self.results['xIndDir']
                self.obj = self.results['obj']
                self.budgets = self.results['budgets']
                self.optObj = self.results['optObj']
		try:
			self.prop = self.results['apsDict']
		except:
			self.prop = None		

	def loadDataObj(self,dataFile):
		with open(dataFile) as f:
                        self.data = pickle.load(f)


	def plotInstChanges(self,dataObj,instInd,fName,pInstInd):
                '''
                Plot the recommended changes of an instance by budget value
		-- pInstInd is the index value when there are only positive indices in the 
		results change file.
                '''
		xLab = r'Feature Change'#Latex
		yLab = r'Budget' #Latex
		title = r'Recommended Treatment Policy vs. Budget for Student '+str(instInd)
		#title = r'Average Actual vs Average Predicted Probability: No Geo Feats' #Latex
	

		lsList = ['-','--','-.',':']

		colList = ['r', 'g', 'b', 'y','c','m','k']

		lWidth = 2.0
		axFntSz = 12
		tiFntSz = 12
	

		stdMod = .25

		#Figure
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.axis([0,len(self.budgets), -1.0, 1.0])
		#ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']) +
                #                        cycler('linestyle', lsList))

		colLin = np.array(list(product(colList,lsList)))


		## Setup latex stuff ##
		#plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		instInit = dataObj['dSet2'][pInstInd,dataObj['changeableIndex']-1]
		chgVals = np.zeros((len(self.budgets)+1,len(instInit[0])))
		budCount = 0
		for i in self.budgets:
			budCount+=1
			instChg = np.hstack((dataObj['dSet2'][instInd,dataObj['unchangeableIndex']-1],self.xVals[i][pInstInd,:][np.newaxis]))
			instChg = instChg[0][dataObj['changeableIndex']-1]
			chgVals[budCount,:] = (instChg - instInit)[0]

		nonZ = np.nonzero(np.sum(np.absolute(chgVals),axis=0))
		chgVals = chgVals[:,nonZ[0]]
		header = dataObj['header'][0][dataObj['changeableIndex']-1]
		header = header[0][nonZ[0]]
		print("changeVals: "+str(chgVals))
		print("Header vals: "+str(header))
		n,p = chgVals.T.shape
		colLin = colLin[nonZ[0]]
		lineDict = {}
		self.budgets = np.insert(self.budgets,0,0)
		for i in range(n):
			lineDict[i] = ax.plot(self.budgets, chgVals.T[i,:], linewidth=lWidth,label=header[i][0], color = colLin[i][0], 
						linestyle=colLin[i][1])#linestyle=lsList[i%(len(lsList)-1)],linewidth=lWidth,label=header[i][0])
	
		plt.rcParams['axes.linewidth'] = 0.5

		#Set labels and title

		ax.set_xlabel(xLab,fontsize=axFntSz)
		ax.set_ylabel(yLab,fontsize=axFntSz)

		ax.set_title(title,fontsize=tiFntSz)
	
		ax.legend()

		fig.savefig(fName,dpi=300)
		return True	
			


	def computeAvgNZProp(self,dataObj):
		

		header = dataObj['header'][0][dataObj['changeableIndex']-1]
                #print(header)
                #avgChg = np.zeros((len(self.budgets),len(dataObj['changeableIndex'][0])))
                acts= dataObj['dSet2'][:,dataObj['changeableIndex'][0]-1]
                #np.savetxt(sys.stdout, acts, '%5.2f')
                n,p = acts.shape
                #np.savetxt(sys.stdout, recs, '%5.2f')
                #np.savetxt(sys.stdout, diffs, '%5.2f')
                #nonZ = (diffs != 0)

		budCount = -1
		self.avgNZProp = []
		self.budgets.insert(0,0)
                for i in self.budgets:
			print(i)
                        budCount+=1
			if i != 0:
				tmp = np.hstack((dataObj['dSet2'][:,dataObj['unchangeableIndex'][0]-1],self.xVals[i]))
                		recs = tmp[:,dataObj['changeableIndex'][0]-1]
				diffs = recs - acts
				nn,pp = diffs.shape
			else:
				nn,pp = self.prop[i].shape
			#valHolder = []
			rowMean = []
			for k in range(nn):
				if i != 0: 
					vals = diffs[k,:]
					nzIDk = np.nonzero(vals)[0]
					if len(nzIDk) > 0:
						rowMean.append(np.mean(np.nan_to_num(self.prop[i][k,nzIDk])))
				else:
					rowMean.append(np.mean(np.nan_to_num(self.prop[i][k,:])))
			#nzID = np.nonzero(diffs)
			#print(nzID)
			#print(rowMean)
			#rowMean = np.nanmean(np.where(diffs!=0,self.prop[i],np.nan),axis=1)
			#rowMean = np.mean(self.prop[i][nzID],axis=0)
			rowMeanF = [v for v in rowMean if v <= (np.mean(rowMean)+3*(np.std(rowMean)))]
			self.avgNZProp.append(np.mean(rowMeanF))
		#rowMean = np.mean(self.prop[0],axis=1)
		#self.avgNZProp.insert(0,np.mean(rowMean))
		return self.avgNZProp

	def computeAvgEstDev(self,dataObj):



                header = dataObj['header'][0][dataObj['changeableIndex']-1]
                #print(header)
                #avgChg = np.zeros((len(self.budgets),len(dataObj['changeableIndex'][0])))
                acts= dataObj['X2_est']
                #np.savetxt(sys.stdout, acts, '%5.2f')
                n,p = acts.shape
                #np.savetxt(sys.stdout, recs, '%5.2f')
                #np.savetxt(sys.stdout, diffs, '%5.2f')
                #nonZ = (diffs != 0)

                budCount = -1
                self.avgNZDiffs = []
                for i in self.budgets:
                        budCount+=1
                        tmp = np.hstack((dataObj['dSet2'][:,dataObj['unchangeableIndex'][0]-1],self.xVals[i]))
                        recs = tmp[:,dataObj['changeableIndex'][0]-1]
                        diffs = recs - acts
                        rowMean = np.nanmean(np.where(diffs!=0,diffs,np.nan),1)
                        self.avgNZDiffs.append(np.mean(rowMean))
                return self.avgNZDiffs               	



        def getTopRecs(self,dataObj,budIndex):
                '''
                Get top recommendations based only on perturbations
                Note: need to make a "postives" only version of this method
		for ARIC.
                '''
		header = dataObj['header'][0][dataObj['changeableIndex']-1]
		budVal = self.budgets[budIndex]
                #avgChg = np.zeros((len(self.budgets),len(dataObj['changeableIndex'][0])))
                acts= dataObj['dSet2'][:,dataObj['changeableIndex'][0]-1]
                #np.savetxt(sys.stdout, acts, '%5.2f')
                n,p = acts.shape
                tmp = np.hstack((dataObj['dSet2'][:,dataObj['unchangeableIndex'][0]-1],self.xVals[budVal]))
                recs = tmp[:,dataObj['changeableIndex'][0]-1]
                #np.savetxt(sys.stdout, recs, '%5.2f')
                diffs = recs - acts
                #np.savetxt(sys.stdout, diffs, '%5.2f')
		nonZ = (diffs != 0)
		common = np.sum(nonZ,axis=0)
		print("##### File: "+str(self.rFile)+" #####")
		print(common)
		print(header[0][np.nonzero(common)])
		print("#############################################")
                return common, header[0][np.nonzero(common)]
		                



	def getAvgRecs(self,dataObj):
		header = dataObj['header'][0][dataObj['changeableIndex']-1]
		print(header)
		avgChg = np.zeros((len(self.budgets),len(dataObj['changeableIndex'][0])))
		budCount = -1
		acts= dataObj['dSet2'][:,dataObj['changeableIndex'][0]-1]
		#np.savetxt(sys.stdout, acts, '%5.2f')
		n,p = acts.shape
		for i in self.budgets:
			budCount+=1
			tmp = np.hstack((dataObj['dSet2'][:,dataObj['unchangeableIndex'][0]-1],self.xVals[i]))
			recs = tmp[:,dataObj['changeableIndex'][0]-1]
			#np.savetxt(sys.stdout, recs, '%5.2f')
			diffs = recs - acts
			#np.savetxt(sys.stdout, diffs, '%5.2f')
			means = np.mean(diffs,axis=0)
			avgChg[budCount,:] = np.mean(diffs,axis=0)
		return avgChg

        def getTopActRecs(self):
                '''
                Get top recommendations for features having values that were initially non-zero
                '''
                None
		


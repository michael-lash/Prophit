import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/anlyz_results')

from Result import Result
import numpy as np
import random

def main():


	fPrimeFile = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--grad.pickle"
	resGC1 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint001-Gnorm.pickle"
	resGC2 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint01-Gnorm.pickle"
	resGC3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint1-Gnorm.pickle"
	resGC4 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam1-Gnorm.pickle"
	resGC5 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam10-Gnorm.pickle"
	resGC6 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam100-Gnorm.pickle"
	resGC7 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam1000-Gnorm.pickle"

	datFile = "/Users/mtlash/Cause/Prophit/data/gaussBenchUpdated.pickle"
	outputPlot = '/Users/mtlash/Cause/Prophit/results/figures/benchExpLamProbNEWL10.png'
	labs = ["f'-opt",'g']
	lams = [.001,.01,.1,1,10,100,1000]
	budInd = 10 #Corresponding, in the case of the benchmark datasets, to a budget=3

	res = Result([fPrimeFile,resGC1,resGC2,resGC3,resGC4,resGC5,resGC6,resGC7],datFile)
	fPrimeAvg,gAvg = res.makeCompLamPlot(outputPlot,lams,labs,budInd)
	print("f'-opt: "+str(fPrimeAvg))
	print("g: "+str(gAvg))
	

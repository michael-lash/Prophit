import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/anlyz_results')

from Result import Result
import numpy as np
import random

def main():


	fPrimeFile = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--grad.pickle"
	resGC1 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint1-new.pickle"
	resGC2 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint3-new.pickle"
	resGC3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint5-new.pickle"
	resGC4 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint7-new.pickle"
	resGC5 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint9-new.pickle"


	datFile = "/Users/mtlash/Cause/Prophit/data/gaussBenchUpdated.pickle"
	outputPlot = '/Users/mtlash/Cause/Prophit/results/figures/benchLinLamProbPlotB2.png'
	labs = ["f'",'g']
	lams = [.1,.3,.5,.7,.9]
	budInd = 2 #Corresponding, in the case of the benchmark datasets, to a budget=2

	res = Result([fPrimeFile,resGC1,resGC2,resGC3,resGC4,resGC5],datFile)
	fPrimeAvg,gAvg = res.makeCompLamPlot(outputPlot,lams,labs,budInd)
	print("fPrime: "+str(fPrimeAvg))
	print("g: "+str(gAvg))
	

import os,sys
sys.path.insert(0,'/Users/mtlash/Cause/Prophit/code/anlyz_results')

from Result import Result
import numpy as np
import random

def main():

	#resFCause1 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--noprobopt.pickle"
	resFCause1 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt.pickle"
	resFCause2 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--grad.pickle"
	#resFCause3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam1-new.pickle"
	#resFCause3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint1-new.pickle"
	#resFCause3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lampoint9-new.pickle"
	#resFCause3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam5-G2.pickle"
	resFCause3 = "/Users/mtlash/Cause/Prophit/results/CAUSEbenchICResults--probopt--aug--lam10-Gnorm.pickle"
	#resFNoCause1 = "/Users/mtlash/Cause/Prophit/results/benchICResults--noprobopt.pickle"
	#resFNoCause1 = "/Users/mtlash/Cause/Prophit/results/NOCAUSbenchICResults--noprobopt--sig.pickle"
	datFile = "/Users/mtlash/Cause/Prophit/data/gaussBenchUpdated.pickle"
	outputPlot = '/Users/mtlash/Cause/Prophit/results/figures/benchAPSPlot.png'
	labs = ["f'-no opt","f'-opt",'g']
	

	res = Result([resFCause1,resFCause2,resFCause3],datFile)
	res.makeAvgAPSPlot(outputPlot,labs)

	

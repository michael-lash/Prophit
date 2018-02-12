

class Logger():

	def __init__(self,logFile):
	
		self.lFile = logFile

	def writeOut(self,data):

		with open(self.lFile,'a') as f:
			f.write(data+"\n")
			f.close()

		return True


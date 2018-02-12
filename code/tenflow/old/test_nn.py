from neural_net import learn_model
import tensorflow as tf
import pandas as pd
import numpy as np


dataDir = "/mnt/cifs/yuqi/"
# File holding X
dataFile = dataDir+"peocessed_data.csv"
# File holding y
labelFile = dataDir+"pat_vector_new.csv"
# Where to save the model
saveMod = "/mnt/cifs/tools/tmpSavMod.ckpt"
# Number of instances we should use
testSize = 1000
# Need to set read chunk size since file is so larger...
chunkSize = 2000
#### Neural Net Params ####
hiddenLayer = [100, 20]
batchSize = round(testSize *.05)
maxIters = 2000


# Read in the data using pandas (easier this way)
chunkCount = 0
for chunk in pd.read_csv(dataFile, delimiter=",",header=None,low_memory=False,chunksize=chunkSize):
#xDat = pd.read_csv(dataFile, delimiter=",",header=None,low_memory=False)
    chunkCount +=1
    xDat = np.array(chunk)
    if chunkCount == 1:
        break

# Convert to a numpy array
#xDat = np.array(xDat)

# Load the output vectors
yDat = pd.read_csv(labelFile, delimiter=",",header=None,low_memory=False)
yDat = np.array(yDat)
ny,py = yDat.shape
# Remove header
yDat = yDat[1:ny,:]

# Get the sample to use for testing...
sampX = xDat[0:testSize,:]
sampY = yDat[0:testSize,:]


#### Train a model ####
trModel,x_d,y_t = learn_model(sampX,sampY,hiddenLayer,batchSize,maxIters,saveMod,bTerm=True)

#### Try using the model to make predictions ####

#Get some data...
evX = xDat[testSize+1:testSize+101,:]
evY = yDat[testSize+1:testSize+101,:]

feedD = {x_d:evX, y_t:evY}

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    #sess.run(tf.variables_initializer([x_d,y_t]))
    sess.run(init_op)
    #temp_loss = sess.run(loss,feed_dict={x_d:evX, y_t:evY})
    preds = sess.run(trModel,feed_dict={x_d:evX})

#print("The accuracy is: "+str(1-temp_loss))
print("The size of the output: "+str(preds.shape))

print("The predictions are: "+str(preds))

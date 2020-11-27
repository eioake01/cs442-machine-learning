import numpy as np

rng = np.random.default_rng()

data = np.loadtxt("data.txt",delimiter=",",usecols=[*range(1,55)],dtype=str)
data = [['-1' if x=="<-1*" else x for x in inputInstance] for inputInstance in data]
data = np.array(data,dtype=float)
rng.shuffle(data)

inputData = data[:,1:55]
data[:,1:55] = inputData / inputData.max(axis=0)


trainData = data[0:21]
testData = data[21:31]


np.savetxt("training.txt",trainData,delimiter=",")
np.savetxt("test.txt",testData,delimiter=",")

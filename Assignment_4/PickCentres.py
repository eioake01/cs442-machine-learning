import numpy as np

inputData = np.loadtxt("training.txt",delimiter=",",usecols=[*range(1,54)])
for x in range(1,inputData.shape[0]+1): 
    centres = inputData[np.random.choice(inputData.shape[0], x, replace=False), :]
    path = "centres"+str(x)+".txt"
    np.savetxt(path,centres,delimiter=",")
import numpy as np


def readData(file):
    inputData = np.loadtxt(file,delimiter=",",usecols=[*range(1,17)],dtype=np.float128)
    outputData = np.loadtxt(file,delimiter=",",usecols=(0),dtype=str)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]


if __name__ == "__main__":  
    data = readData("letter-recognition.txt")
    inputData = data[0]
    outputData = data[1]

    trainInputData = inputData[0:14000, :]
    trainOutputData = outputData[0:14000, :]
    
    testInputData = inputData[14000:20000, :]
    testOutputData = outputData[14000:20000, :]

    gridSize = 5
    maxIterations = 100


    grid =  np.random.rand(gridSize,gridSize)

    for epoch in range(1,maxIterations):
        # Train
        for inputInstance in inputData:
            minDist = float('inf')
            winnerX = 0
            winnerY = 0

            for i in range(gridSize):
                for j in range(gridSize):
                    weight = grid[i][j]
                    sumOfDistances = 0
                    for x in inputData:
                        sumOfDistances += np.square(x-weight)
                        
                    if sumOfDistances < minDist:
                        minDist = sumOfDistances
                        winnerX = i
                        winnerY = j
    

   
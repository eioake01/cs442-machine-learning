import numpy as np


def readData(file):
    inputData = np.loadtxt(file,delimiter=",",usecols=[*range(1,17)],dtype=np.float128)
    outputData = np.loadtxt(file,delimiter=",",usecols=(0),dtype=str)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]

def normalizeInputData(inputData):
    return (inputData - np.min(inputData)) / (np.max(inputData) - np.min(inputData))

def neighbourhoodFunction(s,winnerX,winnerY,X,Y):
    euclideanDistance = np.sqrt(np.square(winnerX-X)+np.square(winnerY-Y))
    return np.exp(-(euclideanDistance/(2*np.square(s))))

if __name__ == "__main__":  
    data = readData("letter-recognition.txt")
    inputData = normalizeInputData(data[0])
    outputData = data[1]

    trainInputData = inputData[0:14000, :]
    trainOutputData = outputData[0:14000, :]
    
    testInputData = inputData[14000:20000, :]
    testOutputData = outputData[14000:20000, :]

    gridSize = 5
    maxIterations = 100

    initialLearningRate = learningRate = 0.1
    initialGaussianWidth = gaussianWidth = gridSize/2

    # Each input has weights to all nodes
    inputWeights =  np.full((16,gridSize,gridSize),np.random())


    for epoch in range(1,maxIterations):
        # Train
        for inputInstance in inputData:
            minDist = float('inf')
            winnerX = 0
            winnerY = 0
        
            # For each node (i,j)
            for i in range(gridSize):
                for j in range(gridSize):
                    sumOfDistances = 0
                    # Calculate sum of distances between each input and the node.
                    for index in range(inputData):
                        x = inputData[index]
                        weight = inputWeights[index][i][j]
                        sumOfDistances += np.square(x-weight)

                    # Find winner
                    if sumOfDistances < minDist:
                        minDist = sumOfDistances
                        winnerX = i
                        winnerY = j
    
            # Update weights
            for i in range(gridSize):
                for j in range(gridSize):
                    for index in range(inputData):
                        h = neighbourhoodFunction(gaussianWidth,winnerX,winnerY,i,j)
                        x = inputData[index]
                        weight = inputWeights[index][i][j] 
                        inputWeights[index][i][j] += learningRate * h * (x-weight)
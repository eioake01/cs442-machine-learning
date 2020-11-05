import numpy as np,math


def readData(file):
    inputData = np.loadtxt(file,delimiter=",",usecols=[*range(1,17)],dtype=np.float128)
    outputData = np.loadtxt(file,delimiter=",",usecols=(0),dtype=str)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]

def normalizeInputData(inputData):
    return (inputData - np.min(inputData)) / (np.max(inputData) - np.min(inputData))

def neighbourhoodFunction(s,winnerX,winnerY,X,Y):
    euclideanDistance = math.sqrt((winnerX-X)**2+(winnerY-Y)**2)
    return math.exp(-(euclideanDistance/(2*(s**2))))

if __name__ == "__main__":  
    data = readData("letter-recognition.txt")
    inputData = normalizeInputData(data[0])
    outputData = data[1]

    trainInputData = inputData[0:14000, :]
    trainOutputData = outputData[0:14000, :]
    
    testInputData = inputData[14000:20000, :]
    testOutputData = outputData[14000:20000, :]

    gridSize = 27
    maxIterations = 100

    initialLearningRate = learningRate = 0.1
    initialGaussianWidth = gaussianWidth = gridSize/2
    denominator = maxIterations/math.log(initialGaussianWidth)
    # Each input has weights to all nodes
    inputWeights =  np.full((16,gridSize,gridSize),np.random())


    for epoch in range(1,maxIterations):
        # Train
        for inputInstance in trainInputData:
            minDist = float('inf')
            winnerX = 0
            winnerY = 0
        
            # For each node (i,j)
            for i in range(gridSize):
                for j in range(gridSize):
                    sumOfDistances = 0
                    # Calculate sum of distances between each input and the node.
                    for index in range(inputInstance):
                        x = inputInstance[index]
                        weight = inputWeights[index][i][j]
                        sumOfDistances += (x-weight) ** 2

                    # Find winner, which node has the least sum of distances between itself and inputs.
                    if sumOfDistances < minDist:
                        minDist = sumOfDistances
                        winnerX = i
                        winnerY = j
    
            # Update weights
            for i in range(gridSize):
                for j in range(gridSize):
                    for index in range(inputInstance):
                        h = neighbourhoodFunction(gaussianWidth,winnerX,winnerY,i,j)
                        x = inputInstance[index]
                        weight = inputWeights[index][i][j] 
                        inputWeights[index][i][j] += learningRate * h * (x-weight)
    
        # At the end of each epoch, update learning rate and gaussian width
        learningRate = initialLearningRate * math.exp(-epoch/maxIterations)
        gaussianWidth = initialGaussianWidth * math.exp(-epoch/denominator)
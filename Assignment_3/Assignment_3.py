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
    maxIterations = 10

    initialLearningRate = learningRate = 0.5
    initialGaussianWidth = gaussianWidth = gridSize/2
    denominator = maxIterations/math.log(initialGaussianWidth)
    error = []

    # Each input has weights to all nodes
    inputWeights =  np.full((16,gridSize,gridSize),np.random.random())


    for epoch in range(maxIterations):
        print(epoch)
        # Train
        trainSumOfMinDistances = 0.0
        winners = np.zeros((trainInputData.shape[0],2))
        for inputInstanceIndex in range(trainInputData.shape[0]):
            minDist = float('inf')
            winnerX = 0
            winnerY = 0       
             
            # For each node (i,j)
            for i in range(gridSize):
                for j in range(gridSize):
                    sumOfDistances = 0
                    # Calculate sum of distances between each input and the node.
                    for inputIndex in range(trainInputData[inputInstanceIndex].shape[0]):
                        x = trainInputData[inputInstanceIndex][inputIndex]
                        weight = inputWeights[inputIndex][i][j]
                        sumOfDistances += (x-weight) ** 2

                    # Find winner, which node has the least sum of distances between itself and inputs.
                    if sumOfDistances < minDist:
                        minDist = sumOfDistances
                        winnerX = i
                        winnerY = j

            trainSumOfMinDistances += minDist
            winners[inputInstanceIndex][0] = winnerX
            winners[inputInstanceIndex][1] = winnerY          

        # Error sum of all min dists square / number of instances
        trainError = (trainSumOfMinDistances ** 2)/trainInputData.shape[0]

    
        # Update weights
        for inputInstanceIndex in range(trainInputData.shape[0]):
            winnerX = winners[inputInstanceIndex][0]
            winnerY = winners[inputInstanceIndex][1]

            for i in range(gridSize):
                for j in range(gridSize):
                    for inputIndex in range(trainInputData[inputInstanceIndex].shape[0]):
                        h = neighbourhoodFunction(gaussianWidth,winnerX,winnerY,i,j)
                        x = trainInputData[inputInstanceIndex][inputIndex]
                        weight = inputWeights[inputIndex][i][j] 
                        inputWeights[inputIndex][i][j] += learningRate * h * (x-weight)

        #Test
        testSumOfMinDistances = 0.0
        for inputInstanceIndex in range(testInputData.shape[0]):
            minDist = float('inf')            

            # For each node (i,j)
            for i in range(gridSize):
                for j in range(gridSize):
                    sumOfDistances = 0
                    # Calculate sum of distances between each input and the node.
                    for inputIndex in range(testInputData[inputInstanceIndex].shape[0]):
                        x = testInputData[inputInstanceIndex][inputIndex]
                        weight = inputWeights[inputIndex][i][j]
                        sumOfDistances += (x-weight) ** 2

                    # Find winner, which node has the least sum of distances between itself and inputs.
                    if sumOfDistances < minDist:
                        minDist = sumOfDistances
                
            testSumOfMinDistances += minDist

        # Error sum of all min dists square / number of instances
        testError = (testSumOfMinDistances ** 2)/testInputData.shape[0]

        error.append([epoch+1,round(trainError,4),round(testError,4)])

        # At the end of each epoch, update learning rate and gaussian width
        learningRate = initialLearningRate * math.exp(-(epoch+1)/maxIterations)
        gaussianWidth = initialGaussianWidth * math.exp(-(epoch+1)/denominator)

    np.savetxt("error.txt",error,fmt='%i %.4f %.4f',delimiter='\t')

    grid = np.empty((gridSize,gridSize),dtype="str_")

    for i in range(gridSize):
        for j in range(gridSize):
            minDist = float('inf')
            minDistIndex = 0

            for inputInstanceIndex in range(testInputData.shape[0]):
                sumOfDistances = 0
                for inputIndex in range(testInputData[inputInstanceIndex].shape[0]):
                    x = testInputData[inputInstanceIndex][inputIndex]
                    weight = inputWeights[inputIndex][i][j]
                    sumOfDistances += (x-weight) ** 2
                if sumOfDistances < minDist:
                    minDist = sumOfDistances
                    minDistIndex = inputInstanceIndex

            grid[i][j] = testOutputData[minDistIndex][0]

    np.savetxt("grid.txt",grid,fmt="%s")

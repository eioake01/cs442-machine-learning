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

    initialLearningRate = learningRate = 0.7
    initialGaussianWidth = gaussianWidth = gridSize/2
    denominator = maxIterations/math.log(initialGaussianWidth)
    error = []

    # Each input has weights to all nodes
    inputWeights =  np.random.rand(16,gridSize,gridSize)

    
    for epoch in range(maxIterations):

        # Train
        trainSumOfMinDistances = 0.0

        # Present train input
        for inputInstanceIndex in range(trainInputData.shape[0]):
            winnerX = 0
            winnerY = 0
            sumOfDistances = 0
            minDist = float('inf')

            
            # For each node
            for i in range(gridSize):
                for j in range(gridSize): 
                        weightsToNode = inputWeights[:,i,j]
                        sumOfDistances = np.sum(np.square(trainInputData[inputInstanceIndex] - weightsToNode))

                        # If smaller than min, assign node as winner
                        if sumOfDistances < minDist:
                            minDist = sumOfDistances
                            winnerX = i
                            winnerY = j

            # After determining winner for this input instance, update weights
            for i in range(gridSize):
                for j in range(gridSize):
                    h = neighbourhoodFunction(gaussianWidth,winnerX,winnerY,i,j)
                    inputWeights[:,i,j] += (learningRate * h * (trainInputData[inputInstanceIndex] -  inputWeights[:,i,j]))

            trainSumOfMinDistances += minDist   

        # Error sum of all min dists square / number of instances
        trainError = (trainSumOfMinDistances ** 2)/trainInputData.shape[0]

        # Test
        testSumOfMinDistances = 0.0

        # Present test input
        for inputInstanceIndex in range(testInputData.shape[0]):
            sumOfDistances = 0
            minDist = float('inf')

            # For each node
            for i in range(gridSize):
                for j in range(gridSize): 

                        weightsToNode = inputWeights[:,i,j]
                        sumOfDistances = np.sum(np.square(testInputData[inputInstanceIndex] - weightsToNode))

                        # If smaller than min, assign node as winner
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

                weightsToNode = inputWeights[:,i,j]
                sumOfDistances = np.sum(np.square(testInputData[inputInstanceIndex] - weightsToNode))

                if sumOfDistances < minDist:
                    minDist = sumOfDistances
                    minDistIndex = inputInstanceIndex

            grid[i][j] = testOutputData[minDistIndex][0]

    np.savetxt("grid.txt",grid,fmt="%s")
import numpy as np,math

def readParameters():
    parameters = open('parameters.txt',"r")

    numInputNeurons = int(parameters.readline().split(" ")[1].strip())
    gridSize = int(parameters.readline().split(" ")[1].strip())
    learningRate = float(parameters.readline().split(" ")[1].strip())
    maxIterations = int(parameters.readline().split(" ")[1].strip())
    dataFile = parameters.readline().split(" ")[1].strip()

    parameters.close()

    return [numInputNeurons,gridSize,learningRate,maxIterations,dataFile]

def readData(file):
    inputData = np.loadtxt(file,delimiter=",",usecols=[*range(1,17)],dtype=np.float128)
    outputData = np.loadtxt(file,delimiter=",",usecols=(0),dtype=str)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]

def normalizeInputData(inputData):
    return (inputData - np.min(inputData)) / (np.max(inputData) - np.min(inputData))

def neighbourhoodFunction(s,winnerX,winnerY,X,Y):
    euclideanDistance = (winnerX-X)**2+(winnerY-Y)**2
    return math.exp(-(euclideanDistance/(2*(s**2))))

def labelling(gridSize,testInputData,inputWeights):
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

    return grid

   

if __name__ == "__main__":  
    parameters = readParameters()
    data = readData(parameters[-1])
    inputData = normalizeInputData(data[0])
    outputData = data[1]

    trainInputData = inputData[0:14000, :]
    trainOutputData = outputData[0:14000, :]
    
    testInputData = inputData[14000:20000, :]
    testOutputData = outputData[14000:20000, :]

    gridSize = parameters[1]
    maxIterations = parameters[3]

    initialLearningRate = learningRate = parameters[2]

    initialGaussianWidth = gaussianWidth = gridSize/2
    denominator = maxIterations/math.log(initialGaussianWidth)
    error = []

    # Each input has weights to all nodes
    inputWeights =  np.array(
        np.random.rand(parameters[0],gridSize,gridSize),
        dtype=np.float128)

    
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


    np.savetxt("results.txt",error,fmt='%i %.4f %.4f',delimiter='\t')

    grid = labelling(gridSize,testInputData,inputWeights)
    np.savetxt("clustering.txt",grid,fmt="%s")

    # Find accuracy before LVQ
    correct = 0
    for inputInstanceIndex in range(testInputData.shape[0]):
        winnerX = 0
        winnerY = 0
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
                        winnerX = i
                        winnerY = j
    
        if(grid[winnerX][winnerY]==testOutputData[inputInstanceIndex]):
            correct += 1             

    print("Accuracy: " + str(correct/testInputData.shape[0]))


    # LVQ
    for inputInstanceIndex in range(testInputData.shape[0]):
            winnerX = 0
            winnerY = 0
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
                            winnerX = i
                            winnerY = j

            # After determining winner check if the label matches the expected output and update weights accordingly.
            if(grid[winnerX][winnerY]==testOutputData[inputInstanceIndex]):                             
                inputWeights[:,winnerX,winnerY] += (learningRate * (testInputData[inputInstanceIndex] -  inputWeights[:,winnerX,winnerY]))       
            else:                                            
                inputWeights[:,winnerX,winnerY] -= (learningRate * (testInputData[inputInstanceIndex] -  inputWeights[:,winnerX,winnerY])) 

    
    LVQgrid = labelling(gridSize,testInputData,inputWeights)
    np.savetxt("LVQclustering.txt",LVQgrid,fmt="%s")

    # Find accuracy after LVQ
    correct = 0
    for inputInstanceIndex in range(testInputData.shape[0]):
        winnerX = 0
        winnerY = 0
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
                        winnerX = i
                        winnerY = j
    
        if(LVQgrid[winnerX][winnerY]==testOutputData[inputInstanceIndex]):
            correct += 1             

    print("Accuracy: " + str(correct/testInputData.shape[0]))  



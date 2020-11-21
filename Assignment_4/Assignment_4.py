import numpy as np,matplotlib.pyplot as plt

def readParameters():
    parameters = open('parameters.txt',"r")

    numHiddenLayerNeurons = int(parameters.readline().split(" ")[1].strip())
    numInputNeurons = int(parameters.readline().split(" ")[1].strip())
    numOutputNeurons = int(parameters.readline().split(" ")[1].strip())
    learningRates = parameters.readline().split(" ")[1].strip()
    sigmas = parameters.readline().split(" ")[1].strip()
    maxIterations = int(parameters.readline().split(" ")[1].strip())
    centresFile = parameters.readline().split(" ")[1].strip()
    trainFile = parameters.readline().split(" ")[1].strip()
    testFile = parameters.readline().split(" ")[1].strip()

    parameters.close()

    return [numInputNeurons,numHiddenLayerNeurons,numOutputNeurons,learningRates,sigmas,maxIterations,centresFile,trainFile,testFile]

def readData(file):
    inputData = np.loadtxt(file,delimiter=",",usecols=[*range(1,54)],dtype=float)
    outputData = np.loadtxt(file,delimiter=",",usecols=(0),dtype=float)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]

def readLearningRates(file):
    return np.loadtxt(file,usecols=(1),dtype=float)

def readSigmas(file):
    return np.loadtxt(file,usecols=(0),dtype=float)

def readCentres(file):
    return np.loadtxt(file,delimiter=",",dtype=float)

def euclideanDistance(inputInstance,centres):
    return np.sum(np.square(inputInstance-centres),axis=1,keepdims=True)

def gaussianFunction(inputInstance,centres,sigmas):
    denominator = 2 * np.square(sigmas)
    return np.exp(-euclideanDistance(inputInstance,centres)/denominator)

def plotError(error):
    error = np.array(error)
    plt.figure(1)
    plt.plot(error[:, 0], error[:, 1], label="Train Error")
    plt.plot(error[:, 0], error[:, 2], label="Test Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error Values")
    plt.title("Error Graph")
    plt.legend()

class RBF:
    def __init__(self,numOfInputNeurons,numHiddenLayerNeurons,numOutputNeurons,learningRates,sigmas,initCentres):
        self.coefficientLearningRate = learningRates[0]
        self.centresLearningRate = learningRates[1]
        self.sigmasLearningRate = learningRates[2]

        self.coefficients = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numHiddenLayerNeurons)).T
        self.biasCoefficient = np.random.uniform(low=-1.0,high=1.0)
        self.centres = initCentres
        self.sigmas = sigmas
        

        self.outputLayer = None

    def forwardPass(self,inputData):
        self.outputLayer =  np.zeros((inputData.shape[0],1))

        for i in range(inputData.shape[0]):
            gaussian = gaussianFunction(inputData[i],self.centres,self.sigmas)
            self.outputLayer[i] = self.biasCoefficient + np.sum(gaussian * self.coefficients)

    def updateVariables(self,inputData,outputData):
        for i in range(inputData.shape[0]):
            inputInstance = inputData[i]

            error = outputData[i] - self.outputLayer[i]
            gaussian = gaussianFunction(inputInstance,self.centres,self.sigmas)
            commonPart = self.coefficients*gaussian

            partOfCentresUpdate = commonPart *((inputInstance-self.centres)/np.square(self.sigmas))
            partOfSigmasUpdate = commonPart * (euclideanDistance(inputInstance,self.centres)/np.power(self.sigmas,3))

            self.coefficients += (self.coefficientLearningRate*error*gaussian)
            self.biasCoefficient += (self.coefficientLearningRate * error)
            self.centres += (self.centresLearningRate * partOfCentresUpdate)
            self.sigmas += (self.sigmasLearningRate*partOfSigmasUpdate)

    def error(self,outputData):
        return np.sum(np.square(self.outputLayer-outputData))/outputData.shape[0]
 


if __name__ == "__main__":  
    parameters = readParameters()
    trainData = readData(parameters[-2])
    testData = readData(parameters[-1])

    trainInputData = trainData[0]
    trainOutputData = trainData[1]

    testInputData = testData[0]
    testOutputData = testData[1]
    
   
    learningRates = readLearningRates(parameters[3])
    sigmas = readSigmas(parameters[4])[np.newaxis].T
    initCentres = readCentres(parameters[6])

    if(initCentres.shape[0]!=parameters[1]):
        print("The number of centres should be the same as the number of neuros of the hidden layer.")
        quit()

    if(sigmas.shape[0]!=parameters[1]):
        print("The number of sigma values should be the same as the number of neuros of the hidden layer.")
        quit()
    

    RBFnetwork = RBF(parameters[0],parameters[1],parameters[2],learningRates,sigmas,initCentres)

    error =  []   
    for epoch in range(parameters[-4]):
        # Forward pass with train data
        RBFnetwork.forwardPass(trainInputData)
        # Calculate train error
        trainError = RBFnetwork.error(trainOutputData)

        # Update variables
        RBFnetwork.updateVariables(trainInputData,trainOutputData)

        # Forward pass with test data
        RBFnetwork.forwardPass(testInputData)
        # Calculate test error
        testError = RBFnetwork.error(testOutputData)
        
        error.append([epoch+1,round(trainError,4),round(testError,4)])


    allCoefficients = np.append(RBFnetwork.coefficients,RBFnetwork.biasCoefficient)

    np.savetxt("error.txt",error,fmt='%i %.4f %.4f',delimiter='\t')
    np.savetxt("coefficients.txt",allCoefficients,fmt="%.4f",delimiter=",")
    plotError(error)
    plt.show()

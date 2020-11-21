import numpy as np

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
    inputData = np.loadtxt(file,delimiter=",",usecols=[*range(1,55)],dtype=float)
    outputData = np.loadtxt(file,delimiter=",",usecols=(0),dtype=str)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]

def readLearningRates(file):
    return np.loadtxt(file,usecols=(1),dtype=float)

def readSigmas(file):
    return np.loadtxt(file,usecols=(0),dtype=float)

def readCentres(file):
    return np.loadtxt(file,usecols=(0),dtype=float)

class RBF:
    def __init__(self,numOfInputNeurons,numHiddenLayerNeurons,numOutputNeurons,learningRates,sigmas,initCentres):
        self.coefficientLearningRate = initLearningRates[0]
        self.centresLearningRate = initLearningRates[1]
        self.sigmasLearningRate = initLearningRates[2]

        self.sigmas = sigmas
        self.centres = initCentres



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
    initCentres = readCentres(parameters[6])[np.newaxis].T

    RBFnetwork = RBF(parameters[0],parameters[1],parameters[2],learningRates,sigmas,initCentres)

import sys,os,numpy as np,math,matplotlib.pyplot as plt

def readParameters():
    parameters = open(os.path.join(sys.path[0],'parameters.txt'),"r")

    numHiddenLayerOneNeurons = int(parameters.readline().split(" ")[1].strip())
    numHiddenLayerTwoNeurons = int(parameters.readline().split(" ")[1].strip())
    numInputNeurons = int(parameters.readline().split(" ")[1].strip())
    numOutputNeurons = int(parameters.readline().split(" ")[1].strip())
    learningRate = float(parameters.readline().split(" ")[1].strip())
    momentum = float(parameters.readline().split(" ")[1].strip())
    maxIterations = int(parameters.readline().split(" ")[1].strip())
    trainFile = parameters.readline().split(" ")[1].strip()
    testFile = parameters.readline().split(" ")[1].strip()

    parameters.close()

    return [numInputNeurons,numHiddenLayerOneNeurons,numHiddenLayerTwoNeurons,
    numOutputNeurons,learningRate,momentum,maxIterations,trainFile,testFile]

def readData(file):
    path = os.path.join(sys.path[0],file)
    inputData = np.loadtxt(path,delimiter=" ",usecols=(0,1))
    inputData = np.append(inputData, np.ones((inputData.shape[0], 1)), axis=1) 
    outputData = np.loadtxt(path,delimiter=" ",usecols=(2)).astype(int)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def deltaOutput(output,error):
    deri = np.multiply(output,(1-output))
    return np.multiply(deri,error)

def deltaHidden(activationValues,d,weights):
    sumOfNext = np.matmul(d,weights)
    deri = np.multiply(activationValues,(1-activationValues))
    delta = np.multiply(deri,sumOfNext)
    return delta

def plotError(error):
    error = np.array(error)
    plt.figure(1)
    plt.plot(error[:, 0], error[:, 1], label="Train Error")
    plt.plot(error[:, 0], error[:, 2], label="Test Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error Values")
    plt.title("Error Graph")
    plt.legend()
    
    
def plotSuccessRate(successRate):
    successRate = np.array(successRate)
    plt.figure(2)
    plt.plot(successRate[:, 0], successRate[:, 1], label="Train Success Rate")
    plt.plot(successRate[:, 0], successRate[:, 2], label="Test Success Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Graph")
    plt.legend()
    


class ANN:
    def __init__(self,numOfInputNeurons,numHiddenLayerOneNeurons,numHiddenLayerTwoNeurons,numOutputNeurons,learningRate,momentum):
        self.weights0 = None
        self.weights1 = None
        self.weights2 = None

        self.weights0pre = None
        self.weights1pre = None
        self.weights2pre = None

        self.layer1 = None
        self.layer2 = None
        self.outputLayer = None 

        self.learningRate = learningRate
        self.momentum = momentum
        
        # Need to check for all cases:
        #   No hidden layer (input to output)
        #   1 hidden layer (input+layer+output)
        #   2 hidden layers (input+layer1+layer2+output)


        if numHiddenLayerOneNeurons == 0:
            self.weights0 = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numOfInputNeurons+1))          
            self.weights0pre = np.zeros_like(self.weights0)
        else:
            self.weights0 = np.random.uniform(low=-1.0,high=1.0,size=(numHiddenLayerOneNeurons,numOfInputNeurons+1))            
            self.weights0pre = np.zeros_like(self.weights0)

            if numHiddenLayerTwoNeurons == 0:
                self.weights1 = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numHiddenLayerOneNeurons+1))        
                self.weights1pre = np.zeros_like(self.weights1)
            else:
                self.weights1 = np.random.uniform(low=-1.0,high=1.0,size=(numHiddenLayerTwoNeurons,numHiddenLayerOneNeurons+1))
                self.weights2 = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numHiddenLayerTwoNeurons+1))

                self.weights1pre = np.zeros_like(self.weights1)
                self.weights2pre = np.zeros_like(self.weights2)
    
    def predict(self,inputData):
        self.forwardPass(inputData)
        return self.outputLayer

    def forwardPass(self,inputData):
        if self.weights1 is None:
            self.outputLayer = sigmoid(np.dot(inputData,self.weights0.T))
        else:
            if self.weights2 is None:
                self.layer1 = sigmoid(np.dot(inputData,self.weights0.T))
                self.layer1 = np.append(self.layer1, np.ones((self.layer1.shape[0], 1)), axis=1) 
                self.outputLayer = sigmoid(np.dot(self.layer1,self.weights1.T))
            else:
                self.layer1 = sigmoid(np.dot(inputData,self.weights0.T))
                self.layer1 = np.append(self.layer1, np.ones((self.layer1.shape[0], 1)), axis=1) 
                self.layer2 = sigmoid(np.dot(self.layer1,self.weights1.T))
                self.layer2 = np.append(self.layer2, np.ones((self.layer2.shape[0], 1)), axis=1) 
                self.outputLayer = sigmoid(np.dot(self.layer2,self.weights2.T))

    def backPropagate(self,outputData,inputData):
        error = outputData - self.outputLayer
       

        if self.weights2 is not None:
            deltaOut = deltaOutput(self.outputLayer,error)
            momentumTerm = self.momentum * (self.weights2 - self.weights2pre)
            adjustment = self.learningRate* np.dot(deltaOut.T,self.layer2) + momentumTerm
            self.weights2pre = self.weights2
            self.weights2 += adjustment

            deltaHid2 = deltaHidden(self.layer2,deltaOut,self.weights2)
            deltaHid2 = np.delete(deltaHid2,-1,axis=1)
            momentumTerm = self.momentum * (self.weights1 - self.weights1pre)
            self.weights1pre = self.weights1
            self.weights1 += (self.learningRate * np.dot(deltaHid2.T,self.layer1) + momentumTerm)


            deltaHid1 = deltaHidden(self.layer1,deltaHid2,self.weights1)
            deltaHid1 = np.delete(deltaHid1,-1,axis=1)
            momentumTerm = self.momentum * (self.weights0 - self.weights0pre)
            self.weights0pre = self.weights0
            self.weights0 += (self.learningRate * np.dot(deltaHid1.T,inputData) + momentumTerm)


        elif self.weights1 is not None:
            deltaOut = deltaOutput(self.outputLayer,error)
            momentumTerm = self.momentum * (self.weights1 - self.weights1pre)
            adjustment = self.learningRate* np.dot(deltaOut.T,self.layer1) + momentumTerm
            self.weights1pre = self.weights1
            self.weights1 += adjustment

        
            deltaHid = deltaHidden(self.layer1,deltaOut,self.weights1)
            deltaHid = np.delete(deltaHid,-1,axis=1)
            momentumTerm = self.momentum * (self.weights0 - self.weights0pre)
            adjustment = self.learningRate * np.dot(deltaHid.T,inputData) + momentumTerm
            self.weights0pre = self.weights0
            self.weights0 += adjustment


        else: 
            deltaOut = deltaOutput(self.outputLayer,error)
            momentumTerm = self.momentum * (self.weights0 - self.weights0pre)
            adjustment = self.learningRate* np.dot(deltaOut.T,self.outputLayer) + momentumTerm
            self.weights0pre = self.weights0
            self.weights0 += adjustment


if __name__ == "__main__":    
    parameters = readParameters() 
    artificialNeuralNetwork = ANN(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
    
    trainData = readData(parameters[-2])
    trainInputData = trainData[0]
    trainOutputData = trainData[1]
    
    testData = readData(parameters[-1])
    testInputData = testData[0]
    testOutputData = testData[1]

    error = []
    successRate = []

    for x in range(parameters[-3]):
        artificialNeuralNetwork.forwardPass(trainInputData)
        trainError = (np.sum((trainOutputData-artificialNeuralNetwork.outputLayer)**2))/(2*trainOutputData.shape[1])
        trainSuccessRate = 1 - np.sum(np.abs((trainOutputData - np.round(artificialNeuralNetwork.outputLayer))))/trainOutputData.shape[0]


        artificialNeuralNetwork.backPropagate(trainOutputData,trainInputData)

        artificialNeuralNetwork.predict(testInputData)
        testError = (np.sum((testOutputData-artificialNeuralNetwork.outputLayer)**2))/(2*testOutputData.shape[1])
        testSuccessRate = 1 - np.sum(np.abs((testOutputData - np.round(artificialNeuralNetwork.outputLayer))))/testOutputData.shape[0]

        error.append([x+1,round(trainError,4),round(testError,4)])
        successRate.append([x+1,trainSuccessRate,testSuccessRate])

    np.savetxt("error.txt",error,fmt='%i %.4f %.4f',delimiter='\t')
    np.savetxt("successrate.txt",successRate,fmt='%i %.4f %.4f',delimiter='\t')

    plotError(error)
    plotSuccessRate(successRate)
    plt.show()
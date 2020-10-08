import sys,os,numpy as np,math

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
            self.weights0 = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numOutnumOfInputNeurons))
            inputBiasWeight = np.full((self.weights0.shape[0],1),-1)
            self.weights0 = np.append(self.weights0,inputBiasWeight,axis=1)
          
            self.weights0pre = np.zeros_like(self.weights0)
        else:
            self.weights0 = np.random.uniform(low=-1.0,high=1.0,size=(numHiddenLayerOneNeurons,numOfInputNeurons))
            inputBiasWeight = np.full((self.weights0.shape[0],1),-1)
            self.weights0 = np.append(self.weights0,inputBiasWeight,axis=1)
          
            self.weights0pre = np.zeros_like(self.weights0)

            if numHiddenLayerTwoNeurons == 0:
                self.weights1 = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numHiddenLayerOneNeurons))
                inputBiasWeight = np.full((self.weights1.shape[0],1),-1)
                self.weights1 = np.append(self.weights1,inputBiasWeight,axis=1)
        

                self.weights1pre = np.zeros_like(self.weights1)
            else:
                self.weights1 = np.random.uniform(low=-1.0,high=1.0,size=(numHiddenLayerTwoNeurons,numHiddenLayerOneNeurons))
                inputBiasWeight = np.full((self.weights1.shape[0],1),-1)
                self.weights1 = np.append(self.weights1,inputBiasWeight,axis=1)
                

                self.weights2 = np.random.uniform(low=-1.0,high=1.0,size=(numOutputNeurons,numHiddenLayerOneNeurons))
                inputBiasWeight = np.full((self.weights2.shape[0],1),-1)
                self.weights2 = np.append(self.weights2,inputBiasWeight,axis=1)

                self.weights1pre = np.zeros_like(self.weights1)
                self.weights2pre = np.zeros_like(self.weights2)

    def train(self,inputData,outputData,maxIterations):
        for _ in range(maxIterations):
            self.forwardPass(inputData)
            self.backPropagate(outputData)
    
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

    def backPropagate(self,outputData):
        error = outputData - self.outputLayer
       

        if self.weights2 is not None:
            d_2 = 1 * np.dot(self.outputLayer.T,np.dot((1-self.outputLayer),error.T))
            momentumTerm = self.momentum * (self.weights2 - self.weights2pre)
            adjustment = self.learningRate* np.dot(d_2,self.layer2) + momentumTerm
            self.weights2pre = self.weights2
            self.weights2 += adjustment


            inner = np.inner(self.weights2,d_2) 
            d_1 = 1 * np.dot(self.layer2.T,(1-self.layer2)) * inner
            momentumTerm = self.momentum * (self.weights1 - self.weights1pre)
            self.weights1pre = self.weights1
            self.weights1 += (self.learningRate * np.dot(self.layer2,d_1.T) + momentumTerm)


            inner = np.inner(self.weights1,d_1) 
            d_0 = 1 * np.dot(self.layer1.T,(1-self.layer1)) * inner
            momentumTerm = self.momentum * (self.weights0 - self.weights0pre)
            self.weights0pre = self.weights0
            self.weights0 += (self.learningRate * np.dot(self.layer1,d_0.T) + momentumTerm)


        elif self.weights1 is not None:
            d_1 = 1 * np.dot(self.outputLayer.T,np.dot((1-self.outputLayer),error.T))
            momentumTerm = self.momentum * (self.weights1 - self.weights1pre)
            adjustment = self.learningRate* np.dot(d_1,self.layer1) + momentumTerm
            self.weights1pre = self.weights1
            self.weights1 += adjustment

        
            print(self.weights1)
            print(d_1)
            inner = np.inner(self.weights1.T,d_1.T) 
            d_0 = 1 * np.dot(self.layer1.T,(1-self.layer1)) 
            momentumTerm = self.momentum * (self.weights0 - self.weights0pre)
            self.weights0pre = self.weights0
            self.weights0 += (np.dot(self.learningRate * d_0,inputData.T) + momentumTerm)


        else: 
            d_0 = 1 * np.dot(self.outputLayer.T,error*(1-self.outputLayer))
            momentumTerm = self.momentum * (self.weights0 - self.weights0pre)
            self.weights0pre = self.weights0
            self.weights0 += (self.learningRate * np.dot(self.outputLayer,d_0.T) + momentumTerm)


           



if __name__ == "__main__":    
    parameters = readParameters()   
    artificialNeuralNetwork = ANN(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
    
    trainData = readData(parameters[-2])
    artificialNeuralNetwork.train(trainData[0],trainData[1],parameters[-3])

    # testData = readData(parameters[-1])
    # print(artificialNeuralNetwork.predict(testData[0]))
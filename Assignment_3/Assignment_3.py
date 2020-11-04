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

    gridSize = 5

    map = np.empty((gridSize,gridSize))

    print(inputData)
    print(outputData)
    print(map)
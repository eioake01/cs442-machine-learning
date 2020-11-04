def readData(file):
    path = os.path.join(sys.path[0],file)
    inputData = np.loadtxt(path,delimiter=",",usecols=[*range(1,17)],dtype=np.float128)
    outputData = np.loadtxt(path,delimiter=",",usecols=(0),dtype=str)
    outputData = np.transpose(outputData[np.newaxis])
    return [inputData,outputData]


if __name__ == "__main__":    
    data = readData("letter-recognition.txt")
    inputData = data[0]
    outputData = data[1]
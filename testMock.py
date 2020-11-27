import numpy as np

# def euclideanDistance(inputInstance,centres):
#     return np.sum(np.square(inputInstance-centres),axis=1,keepdims=True)

# def gaussianFunction(inputInstance,centres,sigmas):
#     denominator = 2 * np.square(sigmas)
#     return np.exp(-euclideanDistance(inputInstance,centres)/denominator)

# centres = np.array([
#     [1.0,2.0,3.0],
#     [4.0,5.0,6.0],
#     [1.0,1.0,1.0]
# ])

# inputData = np.array([
#     [1.0,1.0,1.0],

# ])

# outputData = np.array([
#     [4.0],
# ])

# sigmas = np.array(
#     [
#         [1.0],
#         [2.0],
#         [1.0]
#     ]
# )

# coefficients = np.array(
#     [
#         [1.0],
#         [2.0],
#         [1.0]
#     ]
# )

# biasCoefficient = np.random.uniform(low=-1.0,high=1.0)

# coefficientLearningRate = np.array([0.2])
# centresLearningRate = np.array([0.1])
# sigmasLearningRate = np.array([0.3])

# outputLayer =  np.zeros((inputData.shape[0],1))

# for _ in range(10):
#     # For each input instance, calculate output
#     for i in range(inputData.shape[0]):
#         # Calculate gaussian function value
#         gaussian = gaussianFunction(inputData[i],centres,sigmas)
#         # Output = bias coefficient + the sum of the multiplication 
#         # of coefficients by the gaussian of each centre for the specific instance.
#         outputLayer[i] = biasCoefficient + np.sum(gaussian * coefficients)

#     # print(0.5 * np.sum(np.square(outputData-outputLayer)))

#     # For each input instance, update variables
#     for i in range(inputData.shape[0]):
#         # Current instance
#         inputInstance = inputData[i]

#         # Error of current instance
#         error = outputData[i] - outputLayer[i]

#         # Calculate gaussian
#         gaussian = gaussianFunction(inputInstance,centres,sigmas)

#         # Common equation part between centres and sigmas update
#         # If you notice equation 8 and 9 in RBF Equations PDF, the sum is the same
#         # except the last part.
#         commonPart = error * coefficients * gaussian
        
#         # This is the second part of the equation number 8
#         # common part (which is the sum Note: no need for sum as there is only 1 output)
#         #   x_i(p) - R_hi / s^2h 
#         # for each coordinate i of center h
#         # for all centres
#         partOfCentresUpdate = commonPart *((inputInstance-centres)/np.square(sigmas))
        
#         # This is the second part of the equation number 9
#         # common part (which is the sum Note: no need for sum as there is only 1 output)
#         # euclidean distance (x-r_h)/ s^3_h
#         # for all centres (for all h)
#         partOfSigmasUpdate = commonPart * (euclideanDistance(inputInstance,centres)/np.power(sigmas,3))

#         # update coefficients
#         coefficients += (coefficientLearningRate*error*gaussian)

#         #update bias coefficient
#         biasCoefficient += (coefficientLearningRate * error)  

#         #update centres  
#         centres += (centresLearningRate * partOfCentresUpdate)

#         #update sigmas
#         sigmas += (sigmasLearningRate*partOfSigmasUpdate)

test = np.array([
   [1,2,3],
   [5,5,5] 
])

test_normed = test / test.max(axis=0)
print(te)
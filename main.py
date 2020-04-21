#Import Libraries
import numpy as np
from CatRecognizer import *
import h5py

def Preprocess_Dataset():
    #Import Training Data
    training_data = h5py.File("datasets/train_catvnoncat.h5", "r")
    
    training_inputs  = np.array(training_data["train_set_x"][:])
    training_targets = np.array(training_data["train_set_y"][:])
    
    #Import Testing_Data to ensure correct training
    testing_data = h5py.File("datasets/test_catvnoncat.h5", "r")
    
    testing_inputs  = np.array(testing_data["test_set_x"][:])
    testing_targets = np.array(testing_data["test_set_y"][:])

    #Reshape data
    train_y = training_targets.reshape((1, training_targets.shape[0]))
    test_y  = testing_targets.reshape((1, testing_targets.shape[0]))
    
    train_x = training_inputs.reshape(training_inputs.shape[0], -1).T / 255
    test_x  = testing_inputs.reshape(testing_inputs.shape[0], -1).T / 255

    return train_x, train_y, test_x, test_y

def Flexible_Neural_Network(X, Y, dimensions, learning_rate, epochs):
    np.random.seed(1)

    costs = []
    
    print("[+] Number of training example: ", X.shape[1])

    #Initialize Parameters
    parameters = Initialize_DNN(dimensions)
    
    #Train for the number of epochs
    for i in range(0, epochs):
        #Forward Propagate training examples
        caches, LA = FM_ForwardProp(X, parameters)
        
        #Compute the cost
        cost = ComputeCost(LA, Y)

        #Backpropagate the cost
        gradients = FM_BackProp(LA, Y, caches)
        
        #Update the parameters
        parameters = UpdateParameters(parameters, gradients, learning_rate)
        
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    
    return parameters

    
def main():

    '''Number of layers and neurons in each layer are also hyperparameters'''
    dimensions = [12288, 20, 7, 5, 1]    
    
    train_x, train_y, test_x, test_y = Preprocess_Dataset()

    #Hyperparameters
    learning_rate = 0.009
    epochs = 3000

    #Educated_Kedi are learning parameters for the neural network
    Educated_Kedi = Flexible_Neural_Network(train_x, train_y, dimensions,  learning_rate, epochs)

    

if __name__ == '__main__':
    main()




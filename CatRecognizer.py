import numpy as np

def Initialize_DNN(dimensions):
        
    #dimensions = [14443, 56, 8, 1]
    np.random.seed(1)
    parameters = {}

    for l in range(1, len(dimensions)):
        parameters["W" + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1]) / np.sqrt(dimensions[l - 1])

        parameters["b" + str(l)] = np.zeros((dimensions[l], 1))

        #Assertion checks to make sure of correct shapes
        assert(parameters["W" + str(l)].shape == (dimensions[l], dimensions[l - 1]))
        assert(parameters["b" + str(l)].shape == (dimensions[l], 1))
    
    return parameters


def SL_LinearForward(A, W, b):
    Z = W.dot(A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def SL_FProp(A_prev, W, b, activation):

    #Compute Linear Output and Apply Activation
    if activation == "sigmoid":

        Z, linear_cache = SL_LinearForward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = SL_LinearForward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    
    cache = (linear_cache, activation_cache)


    return A, cache





def FM_ForwardProp(inputs, parameters):
    
    A = inputs
    caches = []
    total_layers = len(parameters) // 2

    #Forward Prop L - 1 Layers
    for layer in range(1, total_layers):
        A_prev = A

        #Extract Corresponding Parameters
        W = parameters["W" + str(layer)]
        b = parameters["b" + str(layer)]

        A, cache = SL_FProp(A_prev, W, b, "relu")
        caches.append(cache)
    

    #Forward Prop the Final Layer
    W = parameters["W" + str(total_layers)]
    b = parameters["b" + str(total_layers)]

    AL, cache = SL_FProp(A, W, b, "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, inputs.shape[1]))

    return caches, AL
    



def ComputeCost(AL, Y):
    m = Y.shape[1]
    
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def SL_LBProp(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def SL_ABProp(dA, cache, activation):
    
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = SL_LBProp(dZ, linear_cache)

    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = SL_LBProp(dZ, linear_cache)
      
    
    return dA_prev, dW, db
    



def FM_BackProp(LL_Activation, Y, caches):
    
    gradients = {}
    layers = len(caches)
    m = LL_Activation.shape[1]
    Y = Y.reshape(LL_Activation.shape)

    #Compute Derivative 
    dAL = - (np.divide(Y, LL_Activation) - np.divide(1 - Y, 1 - LL_Activation))

    #Backpropagate the output layer
    current_cache = caches[layers - 1]

    dA_prev, dW, db = SL_ABProp(dAL, current_cache, "sigmoid")

    gradients["dW" + str(layers)]     = dW
    gradients["db" + str(layers)]     = db
    gradients["dA" + str(layers - 1)] = dA_prev

    
    #Backpropagate L - 1 layers and update the gradients dictionary
    for l in reversed(range(layers - 1)):
        current_cache = caches[l]
        dA_prev_tmp, dW_tmp, db_tmp = SL_LBProp(sigmoid_backward(dAL, current_cache[1]), current_cache[0])

        gradients["dW" + str(l + 1)] = dW_tmp
        gradients["db" + str(l + 1)] = db_tmp
        gradients["dA" + str(l)]     = dA_prev_tmp
        
    return gradients



def UpdateParameters(parameters, gradients, learning_rate):
    
    layers = len(parameters) // 2

    for i in range(layers):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * gradients["dW" + str(i + 1)] 
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * gradients["db" + str(i + 1)]

    return parameters



#Activation Functions

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    cache = x
    return s, cache



def relu(Z):
    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)
    
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    Z = cache

    dZ = np.array(dA, copy = True)

    dZ[Z <= 0] = 0

    assert(dZ.shape == Z.shape)


    return dZ

def sigmoid_backward(dA, cache):
    Z = cache

    sigmoid = 1 / (1 + np.exp(-Z))
    dZ = dA * sigmoid * (1 - sigmoid)
    
    assert(dZ.shape == Z.shape)

    return dZ



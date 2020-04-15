#Define the neural class
class neural_container:

    def __init__(self, layer_dims):
        """Initilize the Neural Network"""
        #Variable parameter holds matrices and biases for each layer L
        self.__parameters = self.initilize_deep_layers(layer_dims)

        print ("[+] Deep Neural Network Generation Successful")


    def initilize_deep_layers(self, layer_dims):
        """Initialize Weight and Bias Vectors for Every Layer"""
        np.random.seed(3)

        parameters = {}

        total_layers = len(layer_dims)
        try:
            for l in range(1, total_layers):
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

                assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
                assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        except:
            print ("[!] Problem Occured while generating Deep Network")

        return parameters


    def linear_forward_output(self, A, W, b):
        """Throughputs Linear Calculation for the layer"""
        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def Activate_Linear_Layer(self, A_prev, W, b, activation):
        """Actiavates the linear_forward_output"""

        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward_output(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward_output(A_prev, W, b)
            A, activation_cache = relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def Full_Model_Forward_Prop(self, X, parameters):
        """Propagates the training example through L layers while caching the data required for backprop"""
        caches = []
        A = X
        L = len(parameters) // 2

        for layer in range(1, L):
            A_prev = A
            A, cache = self.Activate_Linear_Layer(A_prev, parameters['W' + str(layer)], parameters['b' + str(layer)], "relu")
            caches.append(cache)

        AL, cache = self.Activate_Linear_Layer(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

    def compute_cost_function(self, AL, Y):
        """Computes the end cost function"""

        m = Y.shape[1]
        cost = - 1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))

        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost

    def Linear_Backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape      == W.shape)
        assert (db.shape      == b.shape)

        return dA_prev, dW, db

    def Linear_Activation_Backward(self, dA, cache, activation):
        """Backward Activates the Layer"""

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)


        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = self.Linear_Backward(dZ, linear_cache)
        return dA_prev, dW, db

    def Full_Model_Backprop(self, AL, Y, caches):
        """Backpropagates and calculates all required gradients"""

        gradients = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        gradients["dA" + str(L - 1)] = self.Linear_Activation_Backward(dAL, current_cache, "sigmoid")[0]
        gradients["dW" + str(L)]     = self.Linear_Activation_Backward(dAL, current_cache, "sigmoid")[1]
        gradients["db" + str(L)]     = self.Linear_Activation_Backward(dAL, current_cache, "sigmoid")[2]


        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.Linear_Backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])

            gradients["dA" + str(l)]     = dA_prev_temp
            gradients["dW" + str(l + 1)] = dW_temp
            gradients["db" + str(l + 1)] = db_temp

        return gradients

    def Update_Parameters(self, parameters, grads, alpha):
        """Updates weights and biases"""

        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] -= alpha * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= alpha * grads["db" + str(l + 1)]

        return parameters



#Main Function
def main():
    deep_network_shape = [5, 4, 3, 7, 9, 100]
    ANN = neural_container(deep_network_shape)

    parameters, grads = update_parameters_test_case()
    parameters = ANN.Update_Parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))



#Initiate main function
if __name__ == '__main__':
    main()

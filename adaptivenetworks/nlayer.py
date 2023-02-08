import numpy as np

telemetry = 1

class nlayer:
    id = 0
    shape = 1               ## Defines self dimension (1D)
    input_layers = []        ## store layer pointer
    weights = None          ## assuming all input activations are concatenated (sorted on layer ID).
    bias = np.array([])      ## store self biases
    activationFn = "linear"         ## store self activation function
    learningRate = 0.05


    ## Caching
    
    #### store last activation as cache to speed up when multiple layers use this layer as input. So this is evaluated only once.
    cachedRun = -2      # runNum when cache was calculated, can be old
    ## cachedRun = -1 & isAdaptive = 0 is for input layers
    cacheValue = None
    
    ## Flag indicating if it was being evaluated.
    #### This can help in case of self loops, when a layer was being evaluated was evaluated again
    #  meaning one of this layer's input_layer has this layer as one of the inputs (called self-loop is a graph).
    #  In this situation, the last cached value of this layer will be returned.
    # this may be used to simulate LSTM Network.
    beingEvaluated = 0  

    ## Error variance
    #### Store absolute sum of errors in terms of array of sum per node in 1D np array

    ## supports changing widths & depths. Not suitable for inputs and outputs
    isDynamic = 1
    


    ## Methods
    def __init__(self, shape=1, inputLayers=[], isInput=0, setInputValues=[], activationFn="linear", isDynamic=0, learningRate=0.05) -> None:
        self.shape = shape
        self.activationFn = activationFn
        self.bias = np.random.rand(shape, 1)
        self.input_layers = []  ## Clearing on reinitializing
        self.learningRate = 0.05

        if(isDynamic==0):
            self.isDynamic = 0
        if(isInput):
            self.cachedRun = -1
            self.isDynamic = 0
            if(len(setInputValues) != 0):
                self.cacheValue = np.array(setInputValues)
        else:
            # generating random weights if given
            if(type(inputLayers) == type([])):
                if(len(inputLayers) != 0):
                    for layer in inputLayers:
                        self.addInputLayer(layer)
            else:
                print("inputLayers should be a List.")
                if(type(inputLayers) == type(nlayer(1))):
                    self.addInputLayer(inputLayers)

    def addInputLayer(self, newInputLayer):
        # check if it doesn't already exists
        for layr in self.input_layers:
            if(newInputLayer == layr):
                print("Layer already exists.")
                return -1

        self.input_layers.append(newInputLayer)
        ## DONE: Generate random weights
        generatedColumn = np.random.rand(self.shape, newInputLayer.shape) - 0.5
        if(type(self.weights) == type(None)):
            self.weights = generatedColumn
        else:
            self.weights = np.concatenate((self.weights, generatedColumn), axis=1)

    def addWidth_to_Layer(self, addWidth, updateShapeValue=1):
        if(addWidth > 0):
            if(updateShapeValue): self.shape += addWidth
            self.bias = np.concatenate((self.bias, (np.random.rand(addWidth) - 0.5)))
            
            ## generating new row of random weights
            generatedRow = np.random.rand(addWidth, self.weights.shape[1]) - 0.5
            self.weights = np.concatenate((self.weights, generatedRow))
        else:
            print("error, doesn't support decrease.")

    def autoCorrectWeights(self, regenerateAll=0):
        if(regenerateAll):
            sumOfInputs = 0
            for layr in self.input_layers:
                sumOfInputs += layr.shape

            self.weights = np.random.rand(self.shape, sumOfInputs) - 0.5
            
        else:
            pass


    def applyActivationFn(self,rawActivation):
        if(self.activationFn == "linear"):
            return rawActivation

        if(self.activationFn == "relu"):
            return np.maximum(rawActivation, 0)

        if(self.activationFn == "softmax"):
            A = np.exp(rawActivation) / sum(np.exp(rawActivation))
            return A
 
    def applyDerivActivationFn(self, input):
        if(self.activationFn == "linear"):
            return 1
        if(self.activationFn == "relu"):
            return (input > 0)
        else:
            if(telemetry): print("Activation Function =", self.activationFn, " didn't match, returning as ReLU")
            return (input > 0)
            
      

    def getActivation(self, runNum):    ## return np array of activation of current layer
        ## beingEvaluated == 1 means the node was triggered by a loop in the network. Returning last value cached prevents infinite loops.

        
        if(self.cachedRun == runNum or self.cachedRun == -1 or self.beingEvaluated == 1):   ## if activation was already calculated for this run OR is an input layer
            if(telemetry): 
                if(self.cachedRun == -1):
                    print("Provided input from cache")
                else:
                    print("Re-used Cached Value")
            return(self.cacheValue)
        else:
            ## compiling a numpy array of all activation values listed in input layer. 
            # inputArr = np.array([])
            inputArr = np.array([[]])

            self.beingEvaluated = 1

            for layrIndx in range(len(self.input_layers)):
                if(inputArr.shape[1] > 0):      ##  Handle first situation when inputArr is empty.
                    inputArr = np.concatenate((inputArr, self.input_layers[layrIndx].getActivation(runNum=runNum)))
                else:
                    inputArr = self.input_layers[layrIndx].getActivation(runNum=runNum)


            self.beingEvaluated = 0

            ##  Checking dimensions of input matrix
            if(len(inputArr.shape) == 1):
                iftelemetry: print("Input values should be a 2D array.")
                inputArr = inputArr[:, np.newaxis]

            ## If weights column have wrong
            if(self.weights.shape[0] != self.shape):
                if(self.weights.shape[0] > self.shape):
                    print("Error weight matrix has less weights then necessary")
                if(self.weights.shape[0] < self.shape):
                    iftelemetry: print("Adding to weight.shape[0]")

                    # Generating new weights for weights matrix but shape value is already present so updateShapeValue = 0
                    self.addWidth_to_Layer(self.shape - self.weights.shape[0], updateShapeValue=0)





            # Checking if shape matches
            if(inputArr.shape[0] > self.weights.shape[1]):
                iftelemetry: print("!!!SHAPE MISMATCH!!!", "inputArr.shape[0] =", inputArr.shape[0], "self.weights.shape[1] =", self.weights.shape[1])
                ## Adjust matrix dimension & adding new random weights to match size
                generatedColumn = np.random.rand(self.weights.shape[0], (inputArr.shape[0] - self.weights.shape[1])) - 0.5
                self.weights = np.concatenate((self.weights, generatedColumn), axis=1)


            elif(inputArr.shape[0] < self.weights.shape[1]):       ## input layer may have been removed causing weight matrix to be larger than inputs
                print("!! Input Layer smaller than expected. !!", "inputArr.shape[0] =", inputArr.shape[0], "self.weights.shape[1] =", self.weights.shape[1])
                return -1
            
            
            rawActivation = np.matmul(self.weights, inputArr) + self.bias
            activation = self.applyActivationFn(rawActivation=rawActivation)

            self.cachedRun = runNum
            # self.cacheValue = activation          ## storing a pointer to activation calculated
            self.cacheValue = np.copy(activation)   ## duplicating array

            iftelemetry: print("activation =", activation, "& cached")  

            return activation


    def correct_error(self, activation_error, runNum):
        # if(type(self.cacheValue)==type(None)):
        #     self.
        if(self.cachedRun >= 0):    ## check if is run before
            ## compiling a numpy array of all activation values listed in input layer. 
            inputArr = np.array([[]])
            self.beingEvaluated = 1
            layerLengths = []   ## Store each layer's length to distribute corrections to them later

            for layrIndx in range(len(self.input_layers)):
                layerLengths.append(self.input_layers[layrIndx].shape)
                if(inputArr.shape[1] > 0):
                    inputArr = np.concatenate((inputArr, self.input_layers[layrIndx].getActivation(runNum=self.runNum)))
                else:
                    inputArr = self.input_layers[layrIndx].getActivation(runNum=runNum)
            self.beingEvaluated = 0


            # inputArr2 = inputArr[np.newaxis]

            batch_size = activation_error.shape[1]

            # if(len(inputArr.shape) == 1):       ## if array is 1D, convert to 2D to support Transpose.
            #     inputArrT = inputArr[np.newaxis].T
            # else:
            #     inputArrT = inputArr.T


            # dZ = self.cacheValue - activation_error
            dZ = activation_error

            dW = (1/batch_size)*np.matmul(dZ, inputArr.T)
            dB = (1/batch_size)*np.sum(dZ)

            oldWeights = self.weights
            ## Updating self weights & biases
            self.weights = self.weights - self.learningRate*dW
            self.bias = self.bias - self.learningRate*dB

            ## Finding errors for input layers
            # dIZ = np.matmul(np.transpose(self.weights),dZ)
            dIZ = np.matmul((oldWeights.T), dZ) * self.applyDerivActivationFn(inputArr)

            ## Splitting input corrections to their corresponding layers
            splitPoints = [0]
            lengthTillNow = 0
            for layerIndx in range(len(layerLengths)):
                lengthTillNow += layerLengths[layerIndx]
                splitPoints.append(lengthTillNow)

                self.input_layers[layerIndx].correct_error(dIZ[splitPoints[-2]:splitPoints[-1]], runNum=runNum)

            return [self.cacheValue]


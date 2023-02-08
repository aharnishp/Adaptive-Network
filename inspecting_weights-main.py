# %%
import numpy as np
import matplotlib 

telementary = 1
runNum = 0      ## Increment to utilise caching
# batch_size=100       ## assumed size of dataset
learningRate = 0.03

# %%
# clear function
# import only system from os
from os import system, name

# import sleep to show output for some time period
from time import sleep

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

# %% [markdown]
# # Layer Class

# %%
class nlayer:
    id = 0
    shape = 1               ## Defines self dimension (1D)
    input_layers = []        ## store layer pointer
    weights = None          ## assuming all input activations are concatenated (sorted on layer ID).
    bias = np.array([])      ## store self biases
    activationFn = "linear"         ## store self activation function


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
    def __init__(self, shape=1, inputLayers=[], isInput=0, setInputValues=[], activationFn="linear", isDynamic=0) -> None:
        self.shape = shape
        self.activationFn = activationFn
        self.bias = np.random.rand(shape, 1)
        self.input_layers = []  ## Clearing on reinitializing

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

    def addWidth_to_Layer(self, addWidth):
        if(addWidth > 0):
            self.shape += addWidth
            self.bias = np.concatenate((self.bias, (np.random.rand(addWidth) - 0.5)))
            
            ## generating new row of random weights
            generatedRow = np.random.rand(addWidth, self.weights.shape[1]) - 0.5
            self.weights = np.concatenate((self.weights, generatedRow))
        else:
            print("error, doesn't support decrease.")


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
            if(telementary): print("Activation Function =", self.activationFn, " didn't match, returning as ReLU")
            return (input > 0)
            
      

    def getActivation(self):    ## return np array of activation of current layer
        ## beingEvaluated == 1 means the node was triggered by a loop in the network. Returning last value cached prevents infinite loops.

        
        if(self.cachedRun == runNum or self.cachedRun == -1 or self.beingEvaluated == 1):   ## if activation was already calculated for this run OR is an input layer
            if(telementary): 
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
                    inputArr = np.concatenate((inputArr, self.input_layers[layrIndx].getActivation()))
                else:
                    inputArr = self.input_layers[layrIndx].getActivation()


            self.beingEvaluated = 0

            ##  Checking dimensions of input matrix
            if(len(inputArr.shape) == 1):
                if(telementary): print("Input values should be a 2D array.")
                inputArr = inputArr[:, np.newaxis]



            # Checking if shape matches
            if(inputArr.shape[0] > self.weights.shape[1]):
                if(telementary): print("!!!SHAPE MISMATCH!!!", "inputArr.shape[0] =", inputArr.shape[0], "self.weights.shape[1] =", self.weights.shape[1])
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

            if(telementary): print("activation =", activation, "& cached")  

            return activation


    def correct_error(self, activation_error):
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
                    inputArr = np.concatenate((inputArr, self.input_layers[layrIndx].getActivation()))
                else:
                    inputArr = self.input_layers[layrIndx].getActivation()
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
            self.weights = self.weights - learningRate*dW
            self.bias = self.bias - learningRate*dB

            ## Finding errors for input layers
            # dIZ = np.matmul(np.transpose(self.weights),dZ)
            dIZ = np.matmul((oldWeights.T), dZ) * self.applyDerivActivationFn(inputArr)

            ## Splitting input corrections to their corresponding layers
            splitPoints = [0]
            lengthTillNow = 0
            for layerIndx in range(len(layerLengths)):
                lengthTillNow += layerLengths[layerIndx]
                splitPoints.append(lengthTillNow)

                self.input_layers[layerIndx].correct_error(dIZ[splitPoints[-2]:splitPoints[-1]])

            return [self.cacheValue]

                



# %% [markdown]
# # Network Class

# %%
class network:
    input_shape=1  # Currently only 1D
    output_shape=1 # Currently only 1D
 
    input_layer = None      ## Pointer to input nlayer
    output_layer = None     ## Pointer to output nlayer

    layers = []
    numberOfLayers = 0      ## used to assign ID to new layer in matrix

    adaptive = 1

    def __init__(self, input_shape, output_shape, insertDefault=0) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Connect output with 1 adaptive neuron input
        self.input_layer = nlayer(input_shape, isInput=1)

        if(insertDefault==1):
            hiddenLayer = nlayer(1,inputLayers=[self.input_layer],activationFn="relu")
            self.output_layer = nlayer(output_shape, inputLayers=[hiddenLayer], isDynamic=1)
        else:
            self.output_layer = nlayer(output_shape, inputLayers=[self.input_layer], isDynamic=1)


    def addLayerAtLast(self, shape, isDynamic=1, activationFn="linear"):
        oldInputs = self.output_layer.input_layers
        newLayer = nlayer(shape=shape, inputLayers=oldInputs, isDynamic=isDynamic, activationFn=activationFn)
        newLayer.weights = self.output_layer.weights
        self.output_layer.weights = None
        self.output_layer.input_layers=[]
        self.output_layer.addInputLayer(newLayer)

        ## Transferring Weight matrix

    def setInput(self, input_values):
        # print("SETTING INPUT LAYER & STORING VALUES")
        if(type(self.input_layer) != type(None)):
            if(len(input_values) < self.input_layer.shape):
                print("ERROR: Unable to reduce input layer shape. Insert len(input values) >= input_shape")
            else:
                self.input_layer.shape = len(input_values)
                self.input_layer.cachedRun = -1
                self.input_layer.isDynamic = 0
                self.input_layer.cacheValue = np.array(input_values)
        else:   ## Initialize new input layer
                self.input_layer = nlayer(len(input_values), isInput=1, setInputValues=np.array(input_values))

                
                linker = self.output_layer
                if(type(linker) != type(None)):
                    while(len(linker.input_layers) > 0):    ## following only oldest (1st in list) links to reach input
                        linker = linker.input_layers[0]                               
                    linker.input_layer = [self.input_layer]



    def forward_prop(self, input_values=None):    # find result activation from input activation and weights
        global runNum
        if(type(input_values) != type(None)):
            self.input_layer.cacheValue = input_values

        if(self.input_layer.cachedRun == -1 and type(self.input_layer.cacheValue) != type(None)):
            output_activations = self.output_layer.getActivation()
            runNum += 1
            return output_activations
        else:
            print("Input uninitialized")
            return -1

    def backward_prop(self, input_values, trueOutput):
        global runNum
        self.input_layer.cacheValue = input_values

        if(telementary): print("getting forward prop predictions")

        predictedOutput = self.output_layer.getActivation()

        if(telementary): print("starting backprop")
        predictions = self.output_layer.correct_error(predictedOutput - trueOutput)
        runNum += 1
        return predictions


# %% [markdown]
# # TESTING

# %% [markdown]
# ## MNIST Dataset Testing

# %%
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('Adaptive-Matrix/mnist-train.csv')
# Adaptive-Matrix/

# %%
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

X_trainT = X_train.T

def one_hot(Y, maxExpected):
    one_hot_Y = np.zeros((Y.size, maxExpected + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

## find the index of most probable number guessed by network
def get_predictions(A2):
    return np.argmax(A2, 0)

## find ratio of correct predictions to all data
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


# %% [markdown]
# #### Testing with non dynamic 784-10-10-10 network

# %%
nt = network(784, 10, insertDefault=0)

# %%
nt.addLayerAtLast(10,isDynamic=0,activationFn="relu")
nt.addLayerAtLast(10,isDynamic=0,activationFn="relu")

nt.output_layer.activationFn = "softmax"

# %%


# %%
telementary = 0
out = (nt.forward_prop(X_train.T[0].T))
telementary = 1
print(out)

# %% [markdown]
# ### Backprop Testing

# %%
telementary = 0

# oldWeights = nt.output_layer.weights

Y_train_oneHot = one_hot(Y_train, maxExpected=9)
predictions = nt.backward_prop(input_values=X_train, trueOutput=Y_train_oneHot)

# newWeights = nt.output_layer.weights

# diffWeight = newWeights - oldWeights

# print(diffWeight)

# %% [markdown]
# ### Gradient Descent on MNIST



# %%
# accuracySum = 0
# runCount = 0

telementary = 0

for it in range(450):
    Y_train_oneHot = one_hot(Y_train, maxExpected=9)
    predictedRAW = nt.backward_prop(input_values=X_train, trueOutput=Y_train_oneHot)[0]

    if(it % 5 == 0):
        pauser = input("Press enter for next iteration.")
        clear()
        print("iterations =", it)
        predictions = get_predictions(predictedRAW)
        print("Accuracy =", get_accuracy(predictions, Y_train))
        print("Weights of output:")
        print(nt.output_layer.weights)
        
    

telementary = 1

# # accuracy = accuracySum/len(X_train)
# accuracy = accuracySum/runCount



# %% [markdown]
# #### Testing with Dynamic Netwok 784-1-10

# %%
# n1 = network(784, 10, insertDefault=1)
# # n1.addLayerAtLast(1,1,"relu")
# n1.output_layer.activationFn = "softmax"

# # %%
# telementary = 0

# # %%
# print(n1.output_layer.weights.shape)
# print(n1.output_layer.input_layers[0].weights.shape)

# # %%
# out = (n1.forward_prop(X_train.T[0].T))
# print(out)

# # %%
# predRaw = n1.backward_prop(input_values=X_train.T[0:1].T, trueOutput=one_hot(Y_train[0:1], maxExpected=9))

# # %%
# telementary = 0

# for it in range(250):
#     Y_train_oneHot = one_hot(Y_train, maxExpected=9)
#     predictedRAW = n1.backward_prop(input_values=X_train, trueOutput=Y_train_oneHot)[0]

#     if(it % 10 == 0):
#         print("iterations =", it)
#         predictions = get_predictions(predictedRAW)
#         print("Accuracy =", get_accuracy(predictions, Y_train))


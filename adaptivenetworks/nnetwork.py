import numpy as np
from .nlayer import nlayer

telemetry = 0

                

class nnetwork:
    input_shape=1  # Currently only 1D
    output_shape=1 # Currently only 1D
 
    input_layer = None      ## Pointer to input nlayer
    output_layer = None     ## Pointer to output nlayer

    layers = []
    numberOfLayers = 0      ## used to assign ID to new layer in matrix

    adaptive = 1

    learningRate = 0.05
    runNum = 0              ## Increment to utilise caching

    def __init__(self, input_shape, output_shape, insertDefault=0, learningRate = 0.05) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Connect output with 1 adaptive neuron input
        self.input_layer = nlayer(input_shape, isInput=1)

        if(insertDefault==1):
            hiddenLayer = nlayer(1,inputLayers=[self.input_layer],activationFn="relu", learningRate=learningRate)
            self.output_layer = nlayer(output_shape, inputLayers=[hiddenLayer], isDynamic=1, learningRate=learningRate)
        else:
            self.output_layer = nlayer(output_shape, inputLayers=[self.input_layer], isDynamic=1, learningRate=learningRate)


    def addLayerAtLast(self, shape, isDynamic=1, activationFn="linear", transferWeights=0, learningRate=learningRate, separationFn=None):
        oldInputs = self.output_layer.input_layers
        newLayer = nlayer(shape=shape, inputLayers=oldInputs, isDynamic=isDynamic, activationFn=activationFn, learningRate=learningRate, separationFn=separationFn)

        if(transferWeights):
            newLayer.weights = self.output_layer.weights
            
        self.output_layer.weights = None
        self.output_layer.input_layers=[]
        self.output_layer.addInputLayer(newLayer)
        self.output_layer.autoCorrectWeights(regenerateAll=0)

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
        # global runNum
        if(type(input_values) != type(None)):
            self.input_layer.cacheValue = input_values

        if(self.input_layer.cachedRun == -1 and type(self.input_layer.cacheValue) != type(None)):
            output_activations = self.output_layer.getActivation(runNum=self.runNum)
            self.runNum += 1
            return output_activations
        else:
            print("Input uninitialized")
            return -1

    def backward_prop(self, input_values, trueOutput):
        # global runNum
        self.runNum += 1
        self.input_layer.cacheValue = input_values

        iftelemetry: print("getting forward prop predictions")

        predictedOutput = self.output_layer.getActivation(runNum=self.runNum)

        iftelemetry: print("starting backprop")
        activError = predictedOutput - trueOutput   ## error in output activations
        predictions = self.output_layer.correct_error(activError, runNum=self.runNum)
        return [predictions, activError]

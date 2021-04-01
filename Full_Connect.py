import numpy
import math
import Activation_Functions
import Config
import random

#layerwise computation of neuron outputs will streamline implementation

class Neuron:

    def __init__(self, bias, activation, aggregation):

        self.bias = bias
        self.activation = activation
        self.aggregation = aggregation

class Layer:

    # create layerwise encodings for weights and neurons respectively. These will then be used to initialise network.

    def __init__(self, num):
        self.neurlist = []
        self.biasvector = []
        self.val_vector = []
        self.num = num

        for neuron in range(self.num):

            bias = random.gauss(Config.InitBiasMean, Config.InitBiasStDev)

            self.neurlist.append(Neuron(bias, 'sigmoid', 'sum'))
            self.biasvector.append(bias)

        self.biasvector = numpy.asarray([self.biasvector])



class WeightArray:

    def __init__(self, dim):

        self.dim = dim

        self.array = numpy.zeros(self.dim)

        for weight in numpy.nditer(self.array, op_flags=['readwrite']):

            weight[...] = random.gauss(Config.InitWeightMean, Config.InitWeightStDev)


def initialise_feedforward(layernum):

    #take list of ints for num of neurons in each layer
    #take number of inputs, outputs
    layergenome = []
    weightgenome = []

    layerindex = 1

    for layer in layernum:

        layergenome.append(Layer(layer))

        #no. columns = no. inputs to initialise weight array for each layer.

        if layerindex<len(layernum):

            dim = (layernum[layerindex],layernum[layerindex-1])

            weightgenome.append(WeightArray(dim))

        layerindex+=1

    modelgenome = (layergenome, weightgenome)

    return modelgenome

class Feedforward:

    def __init__(self, ModelGenome):

        self.LayerGenome = ModelGenome[0]
        self.WeightGenome = ModelGenome[1]

    def NeuronValues(self, InputVector):

        self.LayerGenome[0].val_vector = numpy.transpose(InputVector)

        for layer in range(len(self.LayerGenome)-1):

            self.LayerGenome[layer+1].val_vector = numpy.matmul(self.WeightGenome[layer].array, self.LayerGenome[layer].val_vector)
            #matrix multiplication for weights
            self.LayerGenome[layer+1].val_vector += numpy.transpose(self.LayerGenome[layer+1].biasvector)
            #Adding Bias
            for value in numpy.nditer(self.LayerGenome[layer+1].val_vector, op_flags=['readwrite']):
                value[...]=Activation_Functions.sigmoid(value)
            #Applying activation





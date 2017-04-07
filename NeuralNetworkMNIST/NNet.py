import time
import random
import numpy as np
from utils import *
from transfer_functions import *



class NNet(object):
    
    def __init__(self, n_input, netDims, n_iter=50, learn = 0.1, tf = sigmoid, dtf = dsigmoid):
        """
        netSizes: network sizes array: output size is in position N
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        n_iter: how many n_iter
        learn: initial learning rate
        tf: transfer function
        """
       
        # initialize parameters     
        self.n_iter = n_iter            #n_iter
        self.learn = learn              #learning rate
        self.tf = tf                    #transfer_function
        self.dtf = dtf                  #derivative of transfer_function
        self.n_layers = len(netDims)    #Number of layers, including I/O
        self.netDims = netDims          
        
        # initialize Lists
        self.W = [[] for i in range(self.n_layers)]      #Empty list for arrays of weights
        self.dEdU = [[] for i in range(self.n_layers)]   #Empty list for arrays of partial error derivatives

        # adding extra neurons for bias
        self.n_input = n_input + 1             #Dimension of Input Layer, with bias neuron
        self.netDims = [ x+1 for x in netDims] #Dimension of layers, including output
        self.netDims[-1] -= 1                  #Output layer doesn't need bias neuron

        # set up arrays of 1s for activations
        self.input = np.ones(self.n_input)
        self.values = [np.ones(layer_dim) for layer_dim in self.netDims]
        self.output = self.values[-1]
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.netDims[0] ** (1/2)
        self.W[0] = np.random.normal(loc = 0, scale = input_range, size =(self.n_input, self.netDims[0]-1))
        self.W[1:-1] = [ np.random.uniform(size = (self.netDims[i], self.netDims[i+1]-1))\
             / np.sqrt(self.netDims[i]) for i in range(self.n_layers-2) ]
        self.W[-1] = np.random.uniform(size = (self.netDims[-2], self.netDims[-1])) / np.sqrt(self.netDims[-1])
       
        
    def init_w(self,W):
        self.W = W    # weights
   
    def feedForward(self, inputs):
        # Set input, leaving bias neuron untouched. Compute first activation
        self.input = np.append(inputs, 1.0)
        self.values[0] = np.append(self.tf(np.dot(self.input, self.W[0])), 1.0)

        #Compute  hidden activations, all hidden layers except last one (output)
        for layer in range(1, self.n_layers-1):
            self.values[layer] = np.append(          self.tf( 
                np.dot (self.values[layer-1], self.W[layer]) ) , 1.0 )

        # Compute output activations, without append since there is no bias neuron
        self.values[-1] = self.tf(np.dot(self.values[-2], self.W[-1]))
        self.output = self.values[-1]

        return self.output
    
    def backPropagate(self, targets):
        self.dEdU[-1] = (self.output-targets)*self.dtf(self.output)
        self.dEdU[-2] = np.multiply(np.dot(self.W[-1],self.dEdU[-1]),\
                                       self.dtf(self.values[-2]))
        # calculate error terms for hidden layers
        for layer in range(self.n_layers-2, 0, -1):
            self.dEdU[layer-1] = np.multiply(np.dot(self.W[layer],self.dEdU[layer][:-1]),\
                                           self.dtf(self.values[layer-1]))

        # update network weights
        self.W[-1] -= self.learn * np.outer(self.values[-2], self.dEdU[-1])

        for layer in range(1, self.n_layers-1):
            self.W[layer] -= self.learn * np.outer(self.values[layer-1],self.dEdU[layer][:-1])
        self.W[0] -= self.learn * np.outer(self.input, self.dEdU[0][:-1])

        # calculate error
        E = (1.0/2.0)*sum((targets-self.output)**2.0)
        
        return E
    
    def train(self, data, validation_data, flag_printIter=False, flag_valid=False, flag_plot=False):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
        Validation_accuracies=[]
        
        for it in range(self.n_iter):
            # Shuffle Data
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            error=0.0
            
            # Train iteration: FeedForward + Backpropagation
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
                
            # Measure accuracy    
            Training_accuracies.append(self.predict(data)/len(data)*100)
            if(flag_valid):
                Validation_accuracies.append(self.predict(validation_data)/len(validation_data)*100)
            error=error/len(data)
            errors.append(error)
            if(flag_printIter):
                print("iter: %2d/%2d --> E: %5.10f  -Training_Accuracy:  %2.2f  -t: %2.2f " %(it+1,self.n_iter, error, Training_accuracies[-1], time.time() - start_time))
            
        # Plotting
        if(flag_plot):
            tr_plot = plt.plot(range(1, self.n_iter+1), Training_accuracies, label='training_data')
            val_plot = plt.plot(range(1, self.n_iter+1), Validation_accuracies, label='validation_data')
            plt.xlabel('Nº Iteration')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.show()       
        return Training_accuracies, Validation_accuracies

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count= count 
        return count 
    
    
    
    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'W':self.W}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W=data['W']
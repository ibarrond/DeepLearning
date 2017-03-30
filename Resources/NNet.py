import time
import random
import numpy as np
from Resources.utils import *
import Resources.transfer_functions as tf



class NNet(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, n_iter=50, learning_rate = 0.1):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        n_iter: how many n_iter
        learning_rate: initial learning rate
        """
       
        # initialize parameters
        self.n_iter = n_iter   #n_iter
        self.learning_rate = learning_rate
     
        
        # initialize arrays
        self.input = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden = hidden_layer_size+1 #+1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_out = np.ones(self.output)
        
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.Wi_h = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.Wh_o = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
       
        
    def init_w(self,wi,wo):
        self.Wi_h=wi # weights input -> hidden
        self.Wh_o=wo # weights hidden -> output
   
    def feedForward(self, inputs):
        #Compute input activations
        self.a_input[:-1] = inputs
        
        #Compute hidden activations
        self.a_hidden = np.append(np.dot(self.a_input, self.Wi_h), 1.0)
        self.a_hidden = tf.sigmoid(self.a_hidden)

        # Compute output activations
        self.a_out = np.dot(self.a_hidden, self.Wh_o)
        self.a_out = tf.sigmoid(self.a_out)
        
        return self.a_out
    
    def backPropagate(self, targets):
        # Implement it in the NeuralNetwork.py file and when finalised copy and paste your FeedForward function here
        self.dEdU2 = (self.a_out-targets)*tf.dsigmoid(self.a_out)
        # calculate error terms for hidden
        self.dEdU1 = np.multiply(np.dot(self.Wh_o,self.dEdU2), tf.dsigmoid(self.a_hidden))
        # update output weights
        self.Wh_o = self.Wh_o - self.learning_rate*np.outer(self.a_hidden,self.dEdU2)
        # update input weights
        self.Wi_h = self.Wi_h - self.learning_rate*np.outer(self.a_input,self.dEdU1[:-1])
        # calculate error
        E = (1.0/2.0)*((targets-self.a_out)**2.0)
    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
      
        for it in range(self.n_iter):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            
            error=0.0 
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
            Training_accuracies.append(self.predict(data))
            
            error=error/len(data)
            errors.append(error)
            
           
            print("n_iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.n_iter, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.n_iter+1),errors, "Error")
        plot_curve(range(1,self.n_iter+1), Training_accuracies, "Training_Accuracy")
       
        
     

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
            pickle.dump({'wi':self.W_input_to_hidden, 'wo':self.W_hidden_to_output}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden=data['wi']
        self.W_hidden_to_output = data['wo']
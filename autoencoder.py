import numpy as np
import mlpython.learners.generic as mlgen 
import mlpython.learners.classification as mlclass
import mlpython.mlproblems.generic as mlpb 
import mlpython.mlproblems.classification as mlpbclass
from mlpython.mathutils.nonlinear import sigmoid

class Autoencoder(mlgen.Learner):
    """
    Autoencoder trained with unsupervised learning.

    Option ``lr`` is the learning rate.

    Option ``hidden_size`` is the size of the hidden layer.

    Option ``noise_prob`` is the noise or corruption probability of
    setting each input to 0.

    Option ``seed`` is the seed of the random number generator.
    
    Option ``n_epochs`` number of training epochs.
    """
    
    def __init__(self, 
                 lr=0.1,          # learning rate
                 hidden_size=50,     # hidden layer size
                 noise_prob=0.1,  # probability of setting an input to 0
                 seed=1234,       # seed for random number generator
                 n_epochs=10      # nb. of training iterations
                 ):
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.lr = lr
        self.noise_prob = noise_prob
        self.seed = seed
        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator
    
    def initialize(self,W,b):
        self.W = W
        self.b = b

    def train(self,trainset):
        """
        Train autoencoder for ``self.n_epochs`` iterations.
        Use ``self.W.T`` as the output weight matrix (i.e. use tied weights).
        """
        # Initialize parameters
        input_size = trainset.metadata['input_size']
        self.W = (self.rng.rand(input_size,self.hidden_size)-0.5)/(max(input_size,self.hidden_size))
        self.b = np.zeros((self.hidden_size,))
        self.c = np.zeros((input_size,))
        
        # And gradients.
        self.grad_W = np.zeros((input_size,self.hidden_size))
        self.grad_b = np.zeros((self.hidden_size,))
        self.grad_c = np.zeros((input_size,))
                
        for it in range(self.n_epochs):
            for input,target in trainset:

                # fprop
                new_input = self.apply_noise(input)
                self.h = self.encode(new_input)
                output = self.decode(input_size)
                
                # bprop   
                # Use the real input (not noisy)
                self.grad_c = output - input
                half_grad_w = np.dot(self.grad_c[:,np.newaxis], self.h[:,np.newaxis].T)                
                self.grad_b = np.dot(self.W.T, self.grad_c) * (self.h - self.h**2)
                self.grad_W = half_grad_w + (np.dot(self.grad_b[:,np.newaxis], new_input[:,np.newaxis].T)).T             
                                                
                # Updating the parameters
                self.update()                
            print "Epoch #%d"%it

    def apply_noise(self, input):    
        """
        Change some of the input values to 0
        with probability self.noise_prob.
        Return the noisy output.
        """
        mask = np.random.binomial(1, 1-self.noise_prob, len(input))        
        noisy_input = mask * input
        
        return noisy_input
                    
    def encode(self, input):
        """
        Encode the input vector
        and return the hidden layer.
        """
        h = np.zeros(self.hidden_size)   
       
        preactivation = np.dot(self.W.T, input) + self.b
        sigmoid(preactivation, h)
        
        return h
        
    def decode(self, input_size):
        """
        Decode the hidden layer
        and return the output.
        """
        output = np.zeros(input_size)
        
        preactivation = np.dot(self.W, self.h) + self.c
        sigmoid(preactivation, output)
        
        return output
    
    def update(self):
        """
        Update the weights and biases
        """
        # Update the weight matrix:            
        self.W -= self.lr * self.grad_W  
          
        # Update the bias matrices:
        self.b -= self.lr * np.array(self.grad_b)         
        self.c -= self.lr * np.array(self.grad_c) 
        
    def show_filters(self):
        from matplotlib.pylab import show, draw, ion
        import mlpython.misc.visualize as mlvis
        mlvis.show_filters(0.5*self.W.T,
                           200,
                           16,
                           8,
                           10,20,2)
        show()

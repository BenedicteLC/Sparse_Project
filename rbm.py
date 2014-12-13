import numpy as np
import mlpython.learners.generic as mlgen 
import mlpython.learners.classification as mlclass
import mlpython.mlproblems.generic as mlpb 
import mlpython.mlproblems.classification as mlpbclass
from mlpython.mathutils.nonlinear import sigmoid

class RBM(mlgen.Learner):
    """
    Restricted Boltzmann Machine trained with unsupervised learning.

    Option ``lr`` is the learning rate.

    Option ``hidden_size`` is the size of the hidden layer.

    Option ``CDk`` is the number of Gibbs sampling steps used
    by contrastive divergence.

    Option ``seed`` is the seed of the random number generator.
    
    Option ``n_epochs`` number of training epochs.
    """

    def __init__(self, 
                 lr=0.1,             # learning rate
                 hidden_size=50,    # hidden layer size
                 CDk=1,          # nb. of Gibbs sampling steps
                 seed=1234,      # seed for random number generator
                 n_epochs=10     # nb. of training iterations
                 ):
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.lr = lr
        self.CDk = CDk
        self.seed = seed
        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator

    def initialize(self, W, b):
        self.W = W
        self.b = b

    def encode(self, input):
        
        h_x = np.zeros((self.hidden_size,),dtype=np.double)       
            
        # Forward conditional prob:                
        fwd_preactivation = np.dot(np.transpose(self.W), input) + self.b                
        sigmoid(fwd_preactivation, h_x)
            
        return h_x

    def train(self,trainset):
        """
        Train RBM for ``self.n_epochs`` iterations.
        """
        # Initialize parameters
        input_size = trainset.metadata['input_size']

        # Parameter initialization
        self.W = (self.rng.rand(input_size,self.hidden_size)-0.5)/(max(input_size,self.hidden_size))
        self.b = np.zeros((self.hidden_size,))
        self.c = np.zeros((input_size,))
        
        # And gradients.
        self.grad_W = np.zeros((input_size,self.hidden_size),dtype=np.double)
        self.grad_b = np.zeros((self.hidden_size,),dtype=np.double)
        self.grad_c = np.zeros((input_size,),dtype=np.double)
        
        # And conditional probabilities.
        p_h_given_x = np.zeros((self.hidden_size,),dtype=np.double)
        p_x_given_h = np.zeros((input_size,),dtype=np.double)
        
        self.x_tilde = np.zeros(input_size,dtype=np.double)                   
        
        for it in range(self.n_epochs):
            for input,target in trainset:
                
                ################
                # Perform CD-k
                ################
                self.x_tilde = input
                for i in range(0,self.CDk,1):
                    
                    # Forward conditional prob:                
                    fwd_preactivation = np.dot(np.transpose(self.W), self.x_tilde) + self.b                
                    sigmoid(fwd_preactivation, p_h_given_x)
                    # Compare with uniform distribution:
                    fwd_uniform = np.random.uniform(size=(self.hidden_size,))                    
                    h_x = np.greater(p_h_given_x, fwd_uniform)
                    
                    # Backward conditional prob:
                    bck_preactivation = np.dot(self.W, h_x) + self.c         
                    sigmoid(bck_preactivation, p_x_given_h)                       
                    # Compare with uniform distribution:                 
                    bck_uniform = np.random.uniform(size=(input_size,))
                    self.x_tilde = np.greater(p_x_given_h, bck_uniform)
                    
                ################
                # bprop
                ################ 
                h_x = np.zeros((self.hidden_size,),dtype=np.double)
                h_x_tilde = np.zeros((self.hidden_size,),dtype=np.double)  
                
                fwd_preactivation = np.dot(np.transpose(self.W), input) + self.b
                sigmoid(fwd_preactivation, h_x)
                bck_preactivation = np.dot(np.transpose(self.W), self.x_tilde) + self.b
                sigmoid(bck_preactivation, h_x_tilde)
                
                self.grad_W = np.transpose( np.dot(h_x[:,np.newaxis], np.transpose(input[:,np.newaxis])) \
                                            - np.dot(h_x_tilde[:,np.newaxis], np.transpose(self.x_tilde[:,np.newaxis])) ) 
                self.grad_b = h_x - h_x_tilde                
                self.grad_c = input - self.x_tilde 
                                
                # Updating the parameters
                self.update()        

            print "Epoch #%d"%it
                 
                
    def update(self):
        """
        Update the weights and biases
        """
        # Update the weight matrix:            
        self.W += self.lr * self.grad_W  
          
        # Update the bias matrices:
        self.b += self.lr * np.array(self.grad_b)         
        self.c += self.lr * np.array(self.grad_c)             
                
    def show_filters(self):
        from matplotlib.pylab import show, draw, ion
        import mlpython.misc.visualize as mlvis
        mlvis.show_filters(0.5*self.W.T,
                           200,
                           16,
                           8,
                           10,20,2)
        show()

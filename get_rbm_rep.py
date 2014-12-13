import numpy as np
import os
import sys
import copy
from string import Template
import mlpython.datasets.store as dataset_store
from rbm import RBM
import pickle

experiment_number = 150

def train():

	sys.argv.pop(0);    # Remove first argument

	# Check if every option(s) from parent's script are here.
	if 5 != len(sys.argv):
	    print "Usage: python run_stacked_autoencoders_nnet.py lr hidden_size n_epochs n_cdk seed"
	    print ""
	    print "Ex.: python run_stacked_autoencoders_nnet.py 0.01 50 10 10 1234"
	    sys.exit()

	# Set the constructor
	str_ParamOption = "lr=" + sys.argv[0] + ", " + "hidden_size=" + sys.argv[1] + ", " + "n_epochs=" + sys.argv[2] + ", " +\
		"CDk=" + sys.argv[3] + ", " + "seed=" + sys.argv[4]
	try:
	    objectString = 'myObject = RBM(' + str_ParamOption + ')'
	    exec objectString
	    #code = compile(objectString, '<string>', 'exec')
	    #exec code
	except Exception as inst:
	    print "Error while instantiating RBM (required hyper-parameters are probably missing)"
	    print inst

	print "Loading dataset..."
	trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')
	
	print "Training..."
	myObject.train(trainset)

	#Store the trained dictionary and the parameters to a file.
	pickle.dump((myObject.W, myObject.b, myObject.hidden_size), open("Models/RBM/model%d.pkl"%experiment_number, 'wb'))

def get_representation():
	# Load the dictionary and corresponding args.
	(W, b, hidden_size) = pickle.load(open("Models/RBM/model%d.pkl"%experiment_number,'rb'))

	# Set the constructor
	myObject = RBM(hidden_size=hidden_size)

	print "Loading dataset..."
	trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')

	encoded_trainset = []
	encoded_validset = []
	encoded_testset = []

	print "Initializing..."
	myObject.initialize(W,b)

	print "Encoding the trainset..."
	counter = 0 #Inelegant, I know! I use this to only use the first 1000 values.
	for input,target in trainset:    
		#Encode the sample.
		h = myObject.encode(input)
		encoded_trainset.append(h)

		# counter +=1
		# if counter == 1000:
		#     break

	# Save the datasets to files. 
	filename = "Models/RBM/trainset%d.pkl"%(experiment_number)
	pickle.dump( np.asarray(encoded_trainset) , open(filename, 'wb'))

	counter = 0
	print "Encoding the validset..."
	for input,target in validset:
		#Encode the sample.
		h = myObject.encode(input)
		encoded_validset.append(h)

		# counter +=1
		# if counter == 1000:
		#     break

	filename = "Models/RBM/validset%d.pkl"%(experiment_number)
	pickle.dump( np.asarray(encoded_validset) , open(filename, 'wb'))

	#Note: only need to do it for the best hyper-params at the end.
	# counter = 0
	# print "Encoding the testset..."
	# for input,target in testset:
	#     #Encode the sample.
	#     h = myObject.encode(input)
	#     encoded_testset.append(h)
	    
	#     counter +=1
	#     if counter == 1000:
	#         break

    # filename = "Models/RBM/testset%d.pkl"%(experiment_number)
    # pickle.dump( np.asarray(encoded_testset), open(filename, 'wb'))

# Run...
train()
get_representation()


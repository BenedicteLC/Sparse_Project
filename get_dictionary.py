import numpy as np
import os
import sys
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
from sparsecode import SparseCode
import pickle

experiment_number=8

sys.argv.pop(0);	# Remove first argument

# Check if every option(s) from parent's script are here.
if 5 != len(sys.argv):
    print "Usage: python run_sparse_code.py lr size L1 n_epochs seed"
    print ""
    print "Ex.: python run_sparse_code.py 0.1 20 0.1 5 1234"
    sys.exit()

# Set the constructor
str_ParamOption = "lr=" + sys.argv[0] + ", " + "size=" + sys.argv[1] + ", " + "L1=" + sys.argv[2] + ", " + "n_epochs=" + sys.argv[3] + ", " + "seed=" + sys.argv[4]
str_ParamOptionValue = sys.argv[0] + "\t" + sys.argv[1] + "\t" + sys.argv[2] + "\t" + sys.argv[3] + "\t" + sys.argv[4]
try:
    objectString = 'myObject = SparseCode(' + str_ParamOption + ')'
    exec objectString
    #code = compile(objectString, '<string>', 'exec')
    #exec code
except Exception as inst:
    print "Error while instantiating SparseCode (required hyper-parameters are probably missing)"
    print inst

print "Loading dataset..."
trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')
print "Training..."
myObject.train(trainset)
myObject.show_filters()

#Store the trained dictionary and the parameters to a file.
pickle.dump((myObject.dictionary, myObject.lr, myObject.hidden_size, myObject.L1), open("Models/sparse_model%d.pkl"%experiment_number, 'wb'))

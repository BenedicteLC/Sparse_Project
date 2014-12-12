import numpy as np
import mlpython.datasets.store as dataset_store
from sparsecode import *
import pickle

#NOTE: Always make sure to use the right number.
experiment_number = 1

# Load the dictionary and corresponding args.
(dictionary, lr, hidden_size, L1) = pickle.load(open("Models/sparse_model%d.pkl"%experiment_number,'rb'))

# Set the constructor
str_ParamOption = "lr=" + str(lr) + ", " + "size=" + str(hidden_size) + ", " + "L1=" + str(L1)
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

encoded_trainset = []
trainset_out = []
encoded_validset = []
validset_out = []
encoded_testset = []
testset_out = []

print "Initializing..."
myObject.initialize_dictionary(dictionary)

print "Encoding the trainset..."
counter = 0 #Inelegant, I know! I use this to only use the first 1000 values.
for input,target in trainset:    
    #Run ISTA
    h = myObject.infer(input)
    encoded_trainset.append(h)
    trainset_out.append(target)

    counter +=1
    if counter == 1:
        break

counter = 0
print "Encoding the validset..."
for input,target in validset:
    #Run ISTA
    h = myObject.infer(input)
    encoded_validset.append(h)
    validset_out.append(target)

    counter +=1
    if counter == 1:
        break

# Note: only need to do it for the best hyper-params at the end.
# print "Encoding the testset..."
# for input,target in testset:
#     #Run ISTA
#     h = myObject.infer(input)
#     encoded_testset.append(h)
#     testset_out.append(target)

# Save the datasets to files. We store a tuple of numpy arrays.
filename = "Models/trainset%d.pkl"%(experiment_number)
pickle.dump( (np.asarray(encoded_trainset),np.asarray(trainset_out)) , open(filename, 'wb'))
filename = "Models/validset%d.pkl"%(experiment_number)
pickle.dump( (np.asarray(encoded_validset),np.asarray(validset_out)) , open(filename, 'wb'))
#filename = "Models/testset%d.pkl"%(experiment_number)
#pickle.dump( (np.asarray(encoded_testset),(np.asarray(testset_out)), open("Models/testset%d.pkl"%experiment_number, 'wb'))
import sys
import numpy as np
import mlpython.datasets.store as dataset_store
from sparsecode import *
import pickle

#NOTE: Always make sure to use the right number.
experiment_number = 200

def get_dictionary():
    """
    Train the sparse coding model 
    and save the dictionary and params to
    a file.
    """
    sys.argv.pop(0);    # Remove first argument

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
    #Store the trained dictionary and the parameters to a file.
    pickle.dump((myObject.dictionary, myObject.lr, myObject.hidden_size, myObject.L1), open("Models/SC/dictionary%d.pkl"%experiment_number, 'wb'))

    myObject.show_filters()

def get_representation():

    """
    Grab the dictionary, convert
    the datasets to a sparse representation and
    save them to a file.
    """

    # Load the dictionary and corresponding args.
    (dictionary, lr, hidden_size, L1) = pickle.load(open("Models/SC/dictionary%d.pkl"%experiment_number,'rb'))

    # Set the constructor
    myObject = SparseCode(lr,hidden_size,L1)

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
    #counter = 0 #Inelegant, I know! I use this to only use the first 1000 values.
    for input,target in trainset:    
        #Run ISTA
        h = myObject.infer(input)
        encoded_trainset.append(h)
        trainset_out.append(target)

        # counter +=1
        # if counter == 1000:
        #     break

    # Save the datasets to files. 
    filename = "Models/SC/trainset%d.pkl"%(experiment_number)
    pickle.dump( np.asarray(encoded_trainset) , open(filename, 'wb'))
    filename = "Models/train_outputs.pkl"
    pickle.dump( np.asarray(trainset_out) , open(filename, 'wb'))

    #counter = 0
    print "Encoding the validset..."
    for input,target in validset:
        #Run ISTA
        h = myObject.infer(input)
        encoded_validset.append(h)
        validset_out.append(target)

        # counter +=1
        # if counter == 1000:
        #     break

    filename = "Models/SC/validset%d.pkl"%(experiment_number)
    pickle.dump( np.asarray(encoded_validset) , open(filename, 'wb'))
    filename = "Models/valid_outputs.pkl"
    pickle.dump( np.asarray(validset_out) , open(filename, 'wb'))

    #Note: only need to do it for the best hyper-params at the end.
    # counter = 0
    # print "Encoding the testset..."
    # for input,target in testset:
    #     #Run ISTA
    #     h = myObject.infer(input)
    #     encoded_testset.append(h)
    #     testset_out.append(target)
        
    #     counter +=1
    #     if counter == 1000:
    #         break

    # filename = "Models/SC/testset%d.pkl"%(experiment_number)
    # pickle.dump( np.asarray(encoded_testset), open(filename, 'wb'))
    # filename = "Models/test_outputs.pkl"
    # pickle.dump( np.asarray(testset_out) , open(filename, 'wb'))

# Run...
get_dictionary()
get_representation()
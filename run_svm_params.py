# -*- coding: utf-8 -*-

import pickle
import numpy as np 
from sklearn import svm

experiment_number = 10

print "Loading datasets..."
train_samples = pickle.load(open("Models/SC/trainset%d.pkl"%experiment_number,'rb'))
train_outputs = pickle.load(open("Models/train_outputs.pkl",'rb'))
valid_samples = pickle.load(open("Models/SC/validset%d.pkl"%experiment_number,'rb'))
valid_outputs = pickle.load(open("Models/valid_outputs.pkl",'rb'))

print "Training the svm"
svm = svm.SVC() # Default uses RBF
svm.fit(train_samples, train_outputs)

print "Predicting..."
train_score = svm.score(train_samples,train_outputs)
valid_score = svm.score(valid_samples,valid_outputs)

print "Training accuracy: %.3f, validation accuracy¸: %.3f"%(train_score, valid_score)

#Save the output to file.
with open("Outputs/SC_param_tests.txt", "a") as myfile:
    myfile.write("Experiment %d, training accuracy: %.3f, validation accuracy: %.3f \n"%(experiment_number,train_score, valid_score))



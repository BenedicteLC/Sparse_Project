# -*- coding: utf-8 -*-

import pickle
import numpy as np 
from sklearn import svm

experiment_number = 5
dataset_number = 150

print "Loading datasets..."
train_outputs = pickle.load(open("Models/train_outputs.pkl",'rb'))
valid_outputs = pickle.load(open("Models/valid_outputs.pkl",'rb'))

#sc_train_samples = pickle.load(open("Models/SC/trainset%d.pkl"%dataset_number,'rb'))
#sc_valid_samples = pickle.load(open("Models/SC/validset%d.pkl"%dataset_number,'rb'))
autoe_train_samples = pickle.load(open("Models/AutoE/trainset%d.pkl"%dataset_number,'rb'))
autoe_valid_samples = pickle.load(open("Models/AutoE/validset%d.pkl"%dataset_number,'rb'))
rbm_train_samples = pickle.load(open("Models/RBM/trainset%d.pkl"%dataset_number,'rb'))
rbm_valid_samples = pickle.load(open("Models/RBM/validset%d.pkl"%dataset_number,'rb'))

"""
This is where we combine datasets. First, we begin by training the SVM with
individual datasets, then 2-combinations, then all three. Then, we need a method for combining them
with a fraction from each.
"""
 
# print "Concatenating datasets..."
# concat_train_samples = rbm_train_samples
# concat_valid_samples = rbm_valid_samples
concat_train_samples = np.append(autoe_train_samples,rbm_train_samples,axis=1)
concat_valid_samples = np.append(autoe_valid_samples,rbm_valid_samples,axis=1)

print "Training the svm..."
svm = svm.SVC() # Default uses RBF
svm.fit(concat_train_samples, train_outputs)

print "Predicting..."
train_score = svm.score(concat_train_samples,train_outputs)
valid_score = svm.score(concat_valid_samples,valid_outputs)

print "Training accuracy: %.3f, validation accuracy¸: %.3f"%(train_score, valid_score)

#Save the output to file.
with open("Outputs/Experiments.txt", "a") as myfile:
	myfile.write("Experiment %d, training accuracy: %.3f, validation accuracy: %.3f \n"%(experiment_number,train_score, valid_score))



# -*- coding: utf-8 -*-
"""
ANN_architect belongs to DBPD simulation
Created on April 05

@author: Wang Jialiang
"""



"""
Import packages
"""
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from __Mod_route__ import get_root

rr=get_root()
Task_name='\\20240927'
Task_subname='_mixed_dataset_recog'


"""
Functions
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def norm(x):
    for i in x.shape[2]:
        for j in x.shape[1]:
            pass
    return


"""
Load input
"""
#%%
# Model_r=rr+'\\Models'+Task_name+Task_subname
# Input_neurons_OG = np.load(Model_r+'\\Train_set_DBPD_Input_neurons.npy')
# Labels_OG=np.load(Model_r+'\\Train_set_Labels.npy')
# Input_neurons_test = np.load(Model_r+'\\Test_set_DBPD_Input_neurons.npy')
# Labels_test=np.load(Model_r+'\\Test_set_Labels.npy')

# Labels_OG = Labels_OG.astype(np.int64)
# Labels_test = Labels_test.astype(np.int64)
    #%%


"""
Neural network framework
"""
#%%
# # Normalize the pixel values to be between 0 and 1
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# Define the neural network architecture
input_size=50*50
hidden_size=300
output_size=3
learning_rate=0.01

# Randomly initialize the weights and biases for the hidden and output layers
hidden_weights=np.random.randn(input_size, hidden_size)*np.sqrt(2/input_size)
hidden_weights=hidden_weights/10                                                # to balance large input value
hidden_biases=np.zeros((1, hidden_size))
output_weights=np.random.randn(hidden_size, output_size)*np.sqrt(2/hidden_size)
output_biases=np.zeros((1, output_size))
# # Inherting previous model
# Model_series='\NBPD200_mixed_Gen_5'
# hidden_weights=np.load(rr+'\Models\\'+Task_name+Task_subname+Model_series+'\hidden_weights.npy')
# hidden_biases=np.load(rr+'\Models\\'+Task_name+Task_subname+Model_series+'\hidden_biases.npy')
# output_weights=np.load(rr+'\Models\\'+Task_name+Task_subname+Model_series+'\output_weights.npy')
# output_biases=np.load(rr+'\Models\\'+Task_name+Task_subname+Model_series+'\output_biases.npy')
    #%%


"""
Neural network constructing
"""
#%%
# set training parameters
num_epochs=20
batch_size=180
num_batches=len(Labels_OG)//batch_size
accuracy=pd.DataFrame(index=range(num_epochs+1),columns=['Sce_1','Sce_2_day','Sce_2_night','Sce_2_interference'],dtype='float64')
predict_result=pd.DataFrame(index=range(5000),
                            columns=['Sce_1_labels','Sce_1_predicts',
                                      'Sce_2_day_labels','Sce_2_day_predicts',
                                      'Sce_2_night_labels','Sce_2_night_predicts',
                                      'Sce_2_interference_labels','Sce_2_interference_predicts'])

# Evaluate accuracy of initial neural network (random parameters)
hidden_activations = sigmoid(np.dot(Input_neurons_test[0:5000,:,0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0,0] = np.mean(predicted_labels == Labels_test[0:5000])
print('\n Initial parameters give '+str(accuracy.iloc[0,0]*100)+'% accuracy on Scenario 1 testset')

hidden_activations = sigmoid(np.dot(Input_neurons_test[5074:10074,:,0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0,1] = np.mean(predicted_labels == Labels_test[5074:10074])
print('\n Initial parameters give '+str(accuracy.iloc[0,1]*100)+'% accuracy on Scenario 2 testset (daylight)')

hidden_activations = sigmoid(np.dot(Input_neurons_test[10074:15074,:,0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0,2] = np.mean(predicted_labels == Labels_test[10074:15074])
print('\n Initial parameters give '+str(accuracy.iloc[0,2]*100)+'% accuracy on Scenario 2 testset (nighttime)')

hidden_activations = sigmoid(np.dot(Input_neurons_test[15074:20074,:,0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0,3] = np.mean(predicted_labels == Labels_test[15074:20074])
print('\n Initial parameters give '+str(accuracy.iloc[0,3]*100)+'% accuracy on Scenario 2 testset (interference)')

# training
for epoch in range(num_epochs):
    print('\n Start #'+str(epoch)+' epoch training')
    # Shuffle the training set
    perm=np.random.permutation(len(Labels_OG))
    Labels_train=Labels_OG[perm]
    Input_neurons=Input_neurons_OG[perm]
    
    # Train on mini-batches
    print('\n Mini batch progressing... \n')
    for i in tqdm(range(num_batches)):
        # Select a mini-batch
        start = i * batch_size
        end = start + batch_size
        batch_images = Input_neurons[start:end,:,0]
        batch_labels = Labels_train[start:end]
        
        # Forward pass
        hidden_activations = sigmoid(np.dot(batch_images, hidden_weights) + hidden_biases)
        output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
        
        # Backward pass: pLoss/py
        output_grads = output_activations
        output_grads[np.arange(len(batch_labels)), batch_labels] -= 1           # pL/pz=(pL/py_hat)*(psoftmax(z)/pz) 
        output_grads /= len(batch_labels)
        # Backward pass: pLoss/px
        hidden_grads = np.dot(output_grads, output_weights.T) * hidden_activations * (1 - hidden_activations) # pL/pu=(pL/px_hat)*(psigmoi(h)/ph) 
        
        # Gradient descent
        output_weights -= learning_rate * np.dot(hidden_activations.T, output_grads)
        output_biases -= learning_rate * np.sum(output_grads, axis=0, keepdims=True)
        hidden_weights -= learning_rate * np.dot(batch_images.T, hidden_grads)
        hidden_biases -= learning_rate * np.sum(hidden_grads, axis=0, keepdims=True)

    # Evaluate on the Scenario 1 testset
    hidden_activations = sigmoid(np.dot(Input_neurons_test[0:5000,:,0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_1_labels']=Labels_test[0:5000]
    predict_result['Sce_1_predicts']=predicted_labels
    accuracy.iloc[epoch+1,0] = np.mean(predict_result['Sce_1_labels'] == predict_result['Sce_1_predicts'])
    print('\n Model gained '+str(accuracy.iloc[epoch+1,0]*100)+'% accuracy on Scenario 1 testset after #'+str(epoch+1)+' epoch training')
    
    # Evaluate on the Scenario 2 testset (daylight)
    hidden_activations = sigmoid(np.dot(Input_neurons_test[5074:10074,:,0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_day_labels']=Labels_test[5074:10074]
    predict_result['Sce_2_day_predicts']=predicted_labels
    accuracy.iloc[epoch+1,1] = np.mean(predict_result['Sce_2_day_labels'] == predict_result['Sce_2_day_predicts'])
    print('\n Model gained '+str(accuracy.iloc[epoch+1,1]*100)+'% accuracy on Scenario 2 testset (daylight) after #'+str(epoch+1)+' epoch training')
    
    # Evaluate on the Scenario 2 testset (nighttime)
    hidden_activations = sigmoid(np.dot(Input_neurons_test[10074:15074,:,0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_night_labels']=Labels_test[10074:15074]
    predict_result['Sce_2_night_predicts']=predicted_labels
    accuracy.iloc[epoch+1,2] = np.mean(predict_result['Sce_2_night_labels'] ==  predict_result['Sce_2_night_predicts'])
    print('\n Model gained '+str(accuracy.iloc[epoch+1,2]*100)+'% accuracy on Scenario 2 testset (nighttime) after #'+str(epoch+1)+' epoch training')
        
    # Evaluate on the Scenario 2 testset (interference)
    hidden_activations = sigmoid(np.dot(Input_neurons_test[15074:20074,:,0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_interference_labels']=Labels_test[15074:20074]
    predict_result['Sce_2_interference_predicts']=predicted_labels
    accuracy.iloc[epoch+1,3] = np.mean(predict_result['Sce_2_interference_labels'] == predict_result['Sce_2_interference_predicts'])
    print('\n Model gained '+str(accuracy.iloc[epoch+1,3]*100)+'% accuracy on Scenario 2 testset (interference) after #'+str(epoch+1)+' epoch training')
    
    #%%
    
    
# """
# Export model
# """
# #%%
# Model_series='NBPD200_mixed_Gen_1'
# outdir_models=Model_r+'\\'+Model_series
# if not os.path.exists(outdir_models):
#     os.mkdir(outdir_models)
# np.save(outdir_models+'\output_weights.npy',output_weights)
# np.save(outdir_models+'\output_biases.npy',output_biases)
# np.save(outdir_models+'\hidden_weights.npy',hidden_weights)
# np.save(outdir_models+'\hidden_biases.npy',hidden_biases)
# accuracy.to_csv(outdir_models+'\\Accuracy_evolution.csv')

# outdir_results=rr+'\\Experiments'+Task_name+Task_subname
# if not os.path.exists(outdir_results):
#     os.mkdir(outdir_results)
# predict_result.to_csv(outdir_results+'\\Predict_results_'+Model_series+'.csv')
# accuracy.to_csv(outdir_results+'\\Accuracy_evolution_'+Model_series+'.csv')
    #%%
    

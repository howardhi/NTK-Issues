#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 03:17:50 2023

@author: Anonymous
"""

'''
This is the script to run to replicate the results of our paper. 

By default, this script will place the outputs in the same directory that it 
is located in. 
- For example, if 15d_ntkb1_par.py is located in ".../home/some_directory", then 
so will all of the output files. 

There are two output files from running this script:
    - 15d-ntkb1-boxplot.png
    - 15d-ntkb1.pickle
    
15d-ntkb1-boxplot.png is the boxplot in our paper 
15d-ntkb1.pickle is the file that stores the RMSE information so one 
can access it without having to re-run the experiments. 

One can use the read_pickle.py to generate the table of mean RMSE
along with standard errors.
'''

# %% Import stuff
#from tensorflow.python.keras import backend as K
from parallel_functions import *
from tensorflow.python.keras import backend as K
from sklearn import preprocessing
from joblib import Parallel, delayed
import multiprocessing 
import sklearn 
import pickle
import random
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import initializers as init
from keras import layers
from tensorflow import keras
from keras.datasets import mnist
import scipy.linalg
import pandas as pd
import numpy as np
import math
from tabulate import tabulate
import os
from plotnine import *
import pathlib
#import cupy as cp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K._get_available_gpus()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% Random seed
np.random.seed(11342)
num_cores = multiprocessing.cpu_count()

#%% Setting up filepath and some parameters

plot_path = str(pathlib.Path().absolute()) 

d_in_high = 15
n_obs = 1000
c1 = 1
'''
How do we generate a prediction from GP? 
We should fill in the entire covariance matrix and generate from MVN?
'''

d_resp = 1 # dimension of response

# kernels variance initialization
sigmab02 = 1  
sigmab12 = 1
sigmab22 = 1
sigmaw02 = 1 
sigmaw12 = 1
sigmaw22 = 1

# Additional parameters
theta = 2
c1 = 2
c2 = 2
c_sigma = 2
a_cst = 2
b_cst = 6
# %% Declare n_trials, initialize vectors for storing experimental results

start_time = time.time()
n_trials = 50

# record coefficient of variation for each trial 
gp1_coef_var = np.repeat(0.0, repeats = n_trials)
gp2_coef_var = np.repeat(0.0, repeats = n_trials)
ntkb1_coef_var = np.repeat(0.0, repeats = n_trials)
ntkb2_coef_var = np.repeat(0.0, repeats = n_trials)
ntka1_coef_var = np.repeat(0.0, repeats = n_trials)
ntka2_coef_var = np.repeat(0.0, repeats = n_trials)
ntkj1_coef_var = np.repeat(0.0, repeats = n_trials)
ntkj2_coef_var = np.repeat(0.0, repeats = n_trials)
k1_coef_var = np.repeat(0.0, repeats = n_trials)
y_coef_var = np.repeat(0.0, repeats = n_trials)

# single layer, no penalty. Scores recorded in RMSE (root mean squared error)
gp_1_scores = np.array(np.repeat(0.0, repeats = n_trials))
gp_2_scores = np.array(np.repeat(0.0, repeats = n_trials))
ntkb_1_scores = np.array(np.repeat(0.0, repeats = n_trials))
ntkb_2_scores = np.array(np.repeat(0.0, repeats = n_trials))
ntka_1_scores = np.array(np.repeat(0.0, repeats = n_trials))
ntka_2_scores = np.array(np.repeat(0.0, repeats = n_trials))
ntkj_1_scores = np.array(np.repeat(0.0, repeats = n_trials))
ntkj_2_scores = np.array(np.repeat(0.0, repeats = n_trials))
# non-NTK kernels
k_1_scores = np.array(np.repeat(0.0, repeats = n_trials))

# neural networks
nn_1_scores = np.array(np.repeat(0.0, repeats = n_trials))
nn_2_scores = np.array(np.repeat(0.0, repeats = n_trials))


#%% Data generation hyperparameters

d_in_high = 15 # number of input dimensions
n_obs_sim = 1000 # number of observations

# %% Declare model hyperparameters

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
# neural networks
batch_size = 8
epochs_1 = 3000
epochs_2 = 6000
input_shape = (28**2)
learning_rate = 0.002
n_1_2l = 10000
n_2_2l = 10000
n_1_1l = 10000
d_resp = 1 # dimension of response

# kernels
sigmab02 = 1  
sigmab12 = 1
sigmab22 = 1
sigmaw02 = 1  
sigmaw12 = 1
sigmaw22 = 1


# Additional Parameters
theta = 2
c1 = 2
c2 = 2
c_sigma = 2
a_cst = 2
b_cst = 6


# NN bias/weight initialization
w0_init = keras.initializers.RandomNormal(mean = 0.0, stddev=1)
w1_init = keras.initializers.RandomNormal(mean = 0.0, stddev=1)
w2_init = keras.initializers.RandomNormal(mean = 0.0, stddev=1)

b1_init = keras.initializers.RandomNormal(mean = 0.0, stddev=1)
b2_init = keras.initializers.RandomNormal(mean = 0.0, stddev=1)
# %% Now we enter the for-loop that takes us through the n_trials experiments

for z in range(n_trials):
    print('Currently working on trial # ', z)
    x_values = np.random.uniform(low = -5, high = 7, size = d_in_high * n_obs)
    x_15d = np.reshape(x_values, (n_obs, d_in_high))
    
    # NTK assumption, inputs vectors have unit norm 
    x_sim = preprocessing.normalize(x_15d, norm = 'l2')
    
    # compute kernel/covariance matrix (parallel)
    test_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkb1_ij)
     (data1 = x_sim, data2 = x_sim, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1) 
     for i in range(n_obs) for j in range(i, n_obs))
     ))
    ntkb1_sim_par = np.zeros((n_obs, n_obs))
    test_out_idx = 0
    for i in range(n_obs):
        for j in range(i, n_obs):
            ntkb1_sim_par[i][j] = test_out[test_out_idx]
            ntkb1_sim_par[j][i] = test_out[test_out_idx]
            test_out_idx += 1
    
    '''
    After computing the cov, 
    We generate from MVN of dimension d_in 
    '''
    mean_vec = np.repeat(0, n_obs)
    y_sim = np.random.multivariate_normal(mean_vec, ntkb1_sim_par)

    # release GPU memory
    tf.keras.backend.clear_session()
    
    x_train, x_test, y_train, y_test = train_test_split(x_sim, y_sim,
                                                    test_size = 0.33)
    '''
    At this point in the code we need the following:
    x_train (numpy array)
    x_test (numpy array)
    y_train (numpy array)
    y_test (numpy array)
    '''
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    d_in = x_train.shape[1]

    
    #--------------------------------------------------------------------------
    # Initialize 2 layer NN
    #--------------------------------------------------------------------------
    nn_2 = keras.Sequential(
        [
             keras.Input(shape = d_in),
             layers.Rescaling(scale = 1/math.sqrt(d_in)),
             layers.Dense(units = n_1_2l,
                          activation = 'relu', 
                          kernel_initializer = w0_init,
                          bias_initializer = b1_init,
                          ), 
             layers.Rescaling(scale = 1./math.sqrt(n_1_2l)),
             layers.Dense(units = n_2_2l,
                          activation = 'relu', 
                          kernel_initializer = w1_init,
                          bias_initializer = b1_init,
                          ),
             layers.Rescaling(scale = 1./math.sqrt(n_2_2l)),
             layers.Dense(units = d_resp,
                          kernel_initializer = w2_init,
                          bias_initializer = b2_init,
                          )
         ]
    )
    
    nn_2.compile(loss="mse",
                optimizer=tf.keras.optimizers.SGD(learning_rate = learning_rate),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
                )
    
    nn_2.fit(x_train, y_train, batch_size = batch_size,
            epochs = epochs_2, validation_split = 0.3)
    

    score = nn_2.evaluate(x_test, y_test, verbose = 0)
    nn_2_scores[z] = score[1]
    
    #-----------------------------------------------------------------------------
    # Initialize 1 layer NN
    #-----------------------------------------------------------------------------
    nn_1 = keras.Sequential(
        [
             keras.Input(shape = d_in),
             layers.Rescaling(scale = 1/math.sqrt(d_in)),
             layers.Dense(units = n_1_1l,
                          activation = 'relu', 
                          kernel_initializer = w0_init,
                          bias_initializer = b1_init,
                          ), 
             layers.Rescaling(scale = 1/math.sqrt(n_1_1l)),
             layers.Dense(units = d_resp,
                          kernel_initializer = w2_init,
                          bias_initializer = b2_init,
                          )
         ]
    )
    
    nn_1.compile(loss="mse",
                optimizer=tf.keras.optimizers.SGD(learning_rate = learning_rate),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
                )
    
    nn_1.fit(x_train, y_train, batch_size = batch_size,
            epochs = epochs_1, validation_split = 0.3)
    

    score = nn_1.evaluate(x_test, y_test, verbose = 0)
    nn_1_scores[z] = score[1]
    
    # release GPU memory
    tf.keras.backend.clear_session()
            
    '''Compute kernels using parallelized functions'''
    '''GP1'''
    gp1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_gp1_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, c1=c1, 
      sigmab02 = sigmab02, sigmaw02 = sigmab02,
      sigmab12 = sigmab12, sigmaw12 = sigmaw12) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    GP_k_1 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            GP_k_1[i][j] = gp1_out[test_out_idx]
            GP_k_1[j][i] = gp1_out[test_out_idx]
            test_out_idx += 1
            
    
    '''GP2'''
    gp2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_gp2_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, c1=c1, c2=c2,
      sigmab02 = sigmab02, sigmaw02 = sigmab02,
      sigmab12 = sigmab12, sigmaw12 = sigmaw12,
      sigmab22 = sigmab22, sigmaw22 = sigmaw22) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    
    GP_k_2 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            GP_k_2[i][j] = gp2_out[test_out_idx]
            GP_k_2[j][i] = gp2_out[test_out_idx]
            test_out_idx += 1

    
    '''NTKA1'''
    ntka1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntka1_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, 
      c_sigma = c_sigma) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    
    NTKA_1 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            NTKA_1[i][j] = ntka1_out[test_out_idx]
            NTKA_1[j][i] = ntka1_out[test_out_idx]
            test_out_idx += 1

    
    '''NTKA2'''
    ntka2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntka2_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, 
      c_sigma = c_sigma) 
     for i in range(n_train) for j in range(i, n_train))
     ))  
    
    NTKA_2 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            NTKA_2[i][j] = ntka2_out[test_out_idx]
            NTKA_2[j][i] = ntka2_out[test_out_idx]
            test_out_idx += 1
            
    
    '''NTKB1'''
    ntkb1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkb1_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    
    NTKB_1 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            NTKB_1[i][j] = ntkb1_out[test_out_idx]
            NTKB_1[j][i] = ntkb1_out[test_out_idx]
            test_out_idx += 1
    
    
    '''NTKB2'''
    ntkb2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkb2_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1, c2 = c2, sigmab12 = sigmab12, sigmaw22 = sigmaw22) 
     for i in range(n_train) for j in range(i, n_train))
     ))

    NTKB_2 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            NTKB_2[i][j] = ntkb2_out[test_out_idx]
            NTKB_2[j][i] = ntkb2_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKJ1'''
    ntkj1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkj1_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    
    NTKJ_1 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            NTKJ_1[i][j] = ntkj1_out[test_out_idx]
            NTKJ_1[j][i] = ntkj1_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKJ2'''
    ntkj2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkj2_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12, sigmab12 = sigmab12,
      c1 = c1) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    
    NTKJ_2 = np.zeros((n_train, n_train))

    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            NTKJ_2[i][j] = ntkj2_out[test_out_idx]
            NTKJ_2[j][i] = ntkj2_out[test_out_idx]
            test_out_idx += 1
    
    
    '''K1'''
    laplace_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_laplace_ij)
     (data1 = x_train, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, a_cst = a_cst, b_cst = b_cst) 
     for i in range(n_train) for j in range(i, n_train))
     ))
    K_1 = np.zeros((n_train, n_train))
    test_out_idx = 0
    for i in range(n_train):
        for j in range(i, n_train):
            K_1[i][j] = laplace_out[test_out_idx]
            K_1[j][i] = laplace_out[test_out_idx]
            test_out_idx += 1
    # -------------------------------------------------------------------------
    # Storing inner product with test samples
    # -------------------------------------------------------------------------
    # GP, NTKB shared terms
    H1xy_k = np.zeros(shape=(n_test, n_train))
    H1xy_d_k = np.zeros(shape=(n_test, n_train))
    H2xy_k = np.zeros(shape=(n_test, n_train))
    H2xy_d_k = np.zeros(shape=(n_test, n_train))
    # NTKA terms
    a_H1xy_k = np.zeros(shape=(n_test, n_train))
    a_H1xy_d_k = np.zeros(shape=(n_test, n_train))
    a_H2xy_k = np.zeros(shape=(n_test, n_train))
    a_H2xy_d_k = np.zeros(shape=(n_test, n_train))
    # kernel matrices
    GP_k_1_k = np.zeros(shape=(n_test, n_train))
    GP_k_2_k = np.zeros(shape=(n_test, n_train))
    NTKB_1_k = np.zeros(shape=(n_test, n_train))
    NTKB_2_k = np.zeros(shape=(n_test, n_train))
    NTKA_1_k = np.zeros(shape=(n_test, n_train))
    NTKA_2_k = np.zeros(shape=(n_test, n_train))
    NTKJ_1_k = np.zeros(shape=(n_test, n_train))
    NTKJ_2_k = np.zeros(shape=(n_test, n_train))
    K_1_k = np.zeros(shape=(n_test, n_train))
            
    '''Compute the inner products with test samples'''
    '''GP1_k'''
    k_gp1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_gp1_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, c1=c1, 
      sigmab02 = sigmab02, sigmaw02 = sigmab02,
      sigmab12 = sigmab12, sigmaw12 = sigmaw12) 
     for i in range(n_test) for j in range(n_train))
     ))
    GP_k_1_k_par = np.zeros((n_test, n_train))
    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            GP_k_1_k_par[i][j] = k_gp1_out[test_out_idx]
            test_out_idx += 1
    
    '''GP2_k'''
    k_gp2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_gp2_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, c1=c1, c2=c2,
      sigmab02 = sigmab02, sigmaw02 = sigmab02,
      sigmab12 = sigmab12, sigmaw12 = sigmaw12,
      sigmab22 = sigmab22, sigmaw22 = sigmaw22) 
     for i in range(n_test) for j in range(n_train))
     ))
    
    GP_k_2_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            GP_k_2_k_par[i][j] = k_gp2_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKA1_k'''
    k_ntka1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntka1_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, 
      c_sigma = c_sigma) 
     for i in range(n_test) for j in range(n_train))
     ))
    
    NTKA_1_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            NTKA_1_k_par[i][j] = k_ntka1_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKA2_k'''
    k_ntka2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntka2_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, 
      c_sigma = c_sigma) 
     for i in range(n_test) for j in range(n_train))
     ))  
    
    NTKA_2_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            NTKA_2_k_par[i][j] = k_ntka2_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKB1_k'''
    k_ntkb1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkb1_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1) 
     for i in range(n_test) for j in range(n_train))
     ))
    
    NTKB_1_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            NTKB_1_k_par[i][j] = k_ntkb1_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKB2_k'''
    k_ntkb2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkb2_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1, c2 = c2, sigmab12 = sigmab12, sigmaw22 = sigmaw22) 
     for i in range(n_test) for j in range(n_train))
     ))

    NTKB_2_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            NTKB_2_k_par[i][j] = k_ntkb2_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKJ1_k'''
    k_ntkj1_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkj1_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12,
      c1 = c1) 
     for i in range(n_test) for j in range(n_train))
     ))
    
    NTKJ_1_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            NTKJ_1_k_par[i][j] = k_ntkj1_out[test_out_idx]
            test_out_idx += 1
    
    '''NTKJ2_k'''
    k_ntkj2_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_ntkj2_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, sigmab02 = sigmab02, 
      sigmaw02 = sigmaw02, sigmaw12 = sigmaw12, sigmab12 = sigmab12,
      c1 = c1) 
     for i in range(n_test) for j in range(n_train))
     ))
    
    NTKJ_2_k_par = np.zeros((n_test, n_train))

    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            NTKJ_2_k_par[i][j] = k_ntkj2_out[test_out_idx]
            test_out_idx += 1
    
    '''K1_k'''
    k_laplace_out = np.array((Parallel(n_jobs = num_cores)
     (delayed(compute_laplace_ij)
     (data1 = x_test, data2 = x_train, 
      d_in = d_in_high, i=i, j=j, a_cst = a_cst, b_cst = b_cst) 
     for i in range(n_test) for j in range(n_train))
     ))
    K_1_k_par = np.zeros((n_test, n_train))
    test_out_idx = 0
    for i in range(n_test):
        for j in range(n_train):
            K_1_k_par[i][j] = k_laplace_out[test_out_idx]
            test_out_idx += 1

    # -------------------------------------------------------------------------
    # Compute kernel predictions 
    # -------------------------------------------------------------------------
    # GP (1 layer)
    GP_1_pred = GP_k_1_k_par @ (np.linalg.inv(GP_k_1)) @ y_train
    gp_1_scores[z] = math.sqrt(np.sum(np.square(y_test - GP_1_pred))/n_test)
    # GP (2 layers)
    GP_2_pred = GP_k_2_k_par @ (np.linalg.inv(GP_k_2)) @ y_train
    gp_2_scores[z] = math.sqrt(np.sum(np.square(y_test - GP_2_pred))/n_test)
    # NTKB (1 layers)
    NTKB_1_pred = NTKB_1_k_par @ (np.linalg.inv(NTKB_1)) @ y_train
    ntkb_1_scores[z] = math.sqrt(np.sum(np.square(y_test - NTKB_1_pred))/n_test)
    # NTKB (2 layers)
    NTKB_2_pred = NTKB_2_k_par @ (np.linalg.inv(NTKB_2)) @ y_train
    ntkb_2_scores[z] = math.sqrt(np.sum(np.square(y_test - NTKB_2_pred))/n_test)
    # NTKA (1 layers)
    NTKA_1_pred = NTKA_1_k_par @ (np.linalg.inv(NTKA_1)) @ y_train
    ntka_1_scores[z] = math.sqrt(np.sum(np.square(y_test - NTKA_1_pred))/n_test)
    # NTKA (2 layers)
    NTKA_2_pred = NTKA_2_k_par @ (np.linalg.inv(NTKA_2)) @ y_train
    ntka_2_scores[z] = math.sqrt(np.sum(np.square(y_test - NTKA_2_pred))/n_test)
    # NTKJ (1 layers)
    NTKJ_1_pred = NTKJ_1_k_par @ (np.linalg.inv(NTKJ_1)) @ y_train
    ntkj_1_scores[z] = math.sqrt(np.sum(np.square(y_test - NTKJ_1_pred))/n_test)
    # NTKJ (2 layers)
    NTKJ_2_pred = NTKJ_2_k_par @ (np.linalg.inv(NTKJ_2)) @ y_train
    ntkj_2_scores[z] = math.sqrt(np.sum(np.square(y_test - NTKJ_2_pred))/n_test)
    # K_1
    K_1_pred = K_1_k_par @ (np.linalg.inv(K_1)) @ y_train
    k_1_scores[z] = math.sqrt(np.sum(np.square(y_test - K_1_pred))/n_test)
    
end_time = time.time()

print('Time elapsed: ', end_time - start_time, ' seconds')

#%% Instead of print statements, let's organize those results in a table!

'''
NN results 
'''
nn_1_mean = np.mean(nn_1_scores)
nn_2_mean = np.mean(nn_2_scores)
nn_1_se = np.std(nn_1_scores)/math.sqrt(n_trials)
nn_2_se = np.std(nn_2_scores)/math.sqrt(n_trials)

'''
unregularized results
'''
lapl_mean = np.mean(k_1_scores)
ntkb1_mean = np.mean(ntkb_1_scores)
ntkb2_mean = np.mean(ntkb_2_scores)
ntka1_mean = np.mean(ntka_1_scores)
ntka2_mean = np.mean(ntka_2_scores)
ntkj1_mean = np.mean(ntkj_1_scores)
ntkj2_mean = np.mean(ntkj_2_scores)
gp1_mean = np.mean(gp_1_scores)
gp2_mean = np.mean(gp_2_scores)

lapl_se = np.std(k_1_scores)/math.sqrt(n_trials)
ntka1_se = np.std(ntka_1_scores)/math.sqrt(n_trials)
ntka2_se = np.std(ntka_2_scores)/math.sqrt(n_trials)
ntkj1_se = np.std(ntkj_1_scores)/math.sqrt(n_trials)
ntkj2_se = np.std(ntkj_2_scores)/math.sqrt(n_trials)
ntkb1_se = np.std(ntkb_1_scores)/math.sqrt(n_trials)
ntkb2_se = np.std(ntkb_2_scores)/math.sqrt(n_trials)
gp1_se = np.std(gp_1_scores)/math.sqrt(n_trials)
gp2_se = np.std(gp_2_scores)/math.sqrt(n_trials)


'''
Now let's make a table'
'''

col_names = ['Method', 
             'n_trials', 
             'Mean RMSE',
             'SE of RMSE']

'''table of unregularized results'''
table_data_unreg = [['NN (1 layer)', n_trials, nn_1_mean, nn_1_se],
              ['NN (2 layers)', n_trials, nn_2_mean, nn_2_se],
              ['GP (1 layer)', n_trials, gp1_mean, gp1_se],
              ['GP (2 layer)', n_trials, gp2_mean, gp2_se],
              ['NTKA (1 layer)', n_trials, ntka1_mean, ntka1_se],
              ['NTKA (2 layer)', n_trials, ntka2_mean, ntka2_se],
              ['NTKJ (1 layer)', n_trials, ntkj1_mean, ntkj1_se],
              ['NTKJ (2 layer)', n_trials, ntkj2_mean, ntkj2_se],
              ['NTKB (1 layer)', n_trials, ntkb1_mean, ntkb1_se],
              ['NTKB (2 layer)', n_trials, ntkb2_mean, ntkb2_se],
              ['Laplacian', n_trials, lapl_mean, lapl_se]
              ]

print(tabulate(table_data_unreg, 
               headers = col_names, 
               tablefmt = 'fancy_grid'))

#%% Plot results (unregularized)

'''
Here we will use plotnine's ggplot implementation to make a boxplot of our
results. 
'''

df_gp1 = pd.DataFrame(data = gp_1_scores,
                      columns = ['RMSE'])
df_gp1['method'] = 'GP1'
'''
the above will do vector recycling so every row will have method = GP
'''

df_gp2 = pd.DataFrame(data = gp_2_scores,
                      columns = ['RMSE'])
df_gp2['method'] = 'GP2'

df_ntka1 = pd.DataFrame(data = ntka_1_scores,
                      columns = ['RMSE'])
df_ntka1['method'] = 'NTKA1'

df_ntka2 = pd.DataFrame(data = ntka_2_scores,
                      columns = ['RMSE'])
df_ntka2['method'] = 'NTKA2'

df_ntkj1 = pd.DataFrame(data = ntkj_1_scores,
                      columns = ['RMSE'])
df_ntkj1['method'] = 'NTKJ1'

df_ntkj2 = pd.DataFrame(data = ntkj_2_scores,
                      columns = ['RMSE'])
df_ntkj2['method'] = 'NTKJ2'

df_ntkb1 = pd.DataFrame(data = ntkb_1_scores,
                      columns = ['RMSE'])
df_ntkb1['method'] = 'NTKB1'

df_ntkb2 = pd.DataFrame(data = ntkb_2_scores,
                      columns = ['RMSE'])
df_ntkb2['method'] = 'NTKB2'

df_k1 = pd.DataFrame(data = k_1_scores,
                      columns = ['RMSE'])
df_k1['method'] = 'K1'

df_nn_1 = pd.DataFrame(data = nn_1_scores,
                      columns = ['RMSE'])
df_nn_1['method'] = 'NN1'

df_nn_2 = pd.DataFrame(data = nn_2_scores,
                      columns = ['RMSE'])
df_nn_2['method'] = 'NN2'

df_unreg = pd.concat(
    [df_gp1, df_gp2, 
     df_ntka1, df_ntka2,
     df_ntkj1, df_ntkj2,
     df_ntkb1, df_ntkb2,
     df_k1,
     df_nn_1, df_nn_2],
    axis = 0,
    ignore_index=True
) 
'''axis = 0 is stacking them on top of each other'''

plot_unreg = (
    ggplot(df_unreg, aes(x='method', y='RMSE'))
    + geom_boxplot() 
    + labs(x = 'Method',
           y = 'RMSE', 
           title = 'Prediction task on simulated data')
)
ggplot.save(plot_unreg,
            path= plot_path, 
            filename = '15d-ntkb1-boxplot.png', 
            format='png')

#%% Saving the scores used to generate plots

'''
The idea is that if we ever need to regenerate this plot
then we can simply bring out the scores from a saved file instead of having
to run the code again
'''
plot_variables = {}
plot_variables['df_unreg'] = df_unreg
# plot_variables['df_reg'] = df_reg
with open(plot_path + '/15d-ntkb1.pickle', 'wb') as file:
    pickle.dump(y_coef_var, file)
    pickle.dump(gp1_coef_var, file)
    pickle.dump(gp2_coef_var, file)
    pickle.dump(ntkb1_coef_var, file)
    pickle.dump(ntkb2_coef_var, file)
    pickle.dump(ntka1_coef_var, file)
    pickle.dump(ntka2_coef_var, file)
    pickle.dump(ntkj1_coef_var, file)
    pickle.dump(ntkj2_coef_var, file)
    pickle.dump(plot_variables, file)
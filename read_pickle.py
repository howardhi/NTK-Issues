#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:11:08 2024

@author: Anonymous 
"""

'''This file reads the .pickle output from 15d_ntkb1_par.py
The purpose of the pickle file is to store the RMSE information for future
use without having to re-run the experiments'''
#%% Import
# import pandas as pd 
import pickle
import numpy as np
import math
from tabulate import tabulate
#%% 

'''
The user needs to set the number of trials and the filepath

(The default name for the pickle file is 15d-ntkb1.pickle)
'''
n_trials = 50
path = '.../some_directory/15d-ntkb1.pickle'
'''No need to edit anything below this line'''
objects = []
with (open(path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

#%% populate values of table 

df = objects[9]['df_unreg']

nn_1_scores = np.array(df[df['method'] == 'NN1'].RMSE)
nn_2_scores = np.array(df[df['method'] == 'NN2'].RMSE)

gp_1_scores = np.array(df[df['method'] == 'GP1'].RMSE)
gp_2_scores = np.array(df[df['method'] == 'GP2'].RMSE)

k_1_scores = np.array(df[df['method'] == 'K1'].RMSE)

ntka_1_scores = np.array(df[df['method'] == 'NTKA1'].RMSE)
ntka_2_scores = np.array(df[df['method'] == 'NTKA2'].RMSE)

ntkb_1_scores = np.array(df[df['method'] == 'NTKB1'].RMSE)
ntkb_2_scores = np.array(df[df['method'] == 'NTKB2'].RMSE)

ntkj_1_scores = np.array(df[df['method'] == 'NTKJ1'].RMSE)
ntkj_2_scores = np.array(df[df['method'] == 'NTKJ2'].RMSE)

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


#%% Now let's make a table'


col_names = ['Method', 
             'n_trials', 
             'Mean RMSE',
             'SE of Mean RMSE']

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 07:33:30 2024

@author: Anonymous
"""

'''
---No changes are needed here!---

This file contains the functions that will be called in the computation of

the various kernels used in our experiments.
'''

#%% Import 
import numpy as np
import math

#%% GP1 

def compute_gp1_ij(data1, data2, d_in, i,j, c1, sigmab02, sigmaw02, sigmab12,  sigmaw12):
    '''
    We start with this oddly long function that computes everything, 
    
    then we will comment out the things that are irrelevant to gp1 
    
    then we will keep what remains and test its functionality against the
    
    iterated forloop version
    
    First we compute the entire ij grid. Then we will figure out how to 
    cleverly index so we only compute the upper triangular part (and then 
    simply mirror the results about the diagonal)
    '''
    k0x = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data1[i][:]))
    k0y = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data2[j][:]))
    k0xy = (sigmab02 +
            sigmaw02/(d_in) *
            np.sum(data1[i][:] * data2[j][:]))
    cos_delta_0 = k0xy/(k0x*k0y)
    if(abs(cos_delta_0) > 1):
        delta_0 = 0
    else:
        delta_0 = np.arccos(cos_delta_0)
    h1xy_matrix_ij = (math.sqrt(k0x*k0y)/(2*math.pi) *
                  (math.sin(delta_0)+(math.pi - delta_0)*cos_delta_0))
    
    # GP (1 layer)
    gp_1_matrix_ij = sigmab12 + c1 * sigmaw12 * h1xy_matrix_ij
    
    return gp_1_matrix_ij

#%% GP2

def compute_gp2_ij(data1, data2, d_in, i,j, c1, c2, sigmab02, sigmaw02, sigmab12,  sigmaw12, sigmab22, sigmaw22):
    # Terms for NTKB (1 layer)
    k0x = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data1[i][:]))
    k0y = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data2[j][:]))
    k0xy = (sigmab02 +
            sigmaw02/(d_in) *
            np.sum(data1[i][:] * data2[j][:]))
    cos_delta_0 = k0xy/(k0x*k0y)
    if(abs(cos_delta_0) > 1):
        delta_0 = 0
    else:
        delta_0 = np.arccos(cos_delta_0)
    # Terms for NTKB (2 layers)
    k1x = sigmab12 + c1 * sigmaw12/(2*math.pi) * k0x * math.pi
    k1y = sigmab12 + c1 * sigmaw12/(2*math.pi) * k0y * math.pi
    k1xy = (sigmab12 + 
            c1 * sigmaw12/(2*math.pi) *
            math.sqrt(k0x * k0y) *
            (math.sin(delta_0) + (math.pi - delta_0)*cos_delta_0)
            )
    cos_delta_1 = k1xy/(k1x*k1y)
    if(abs(cos_delta_1) > 1):
        delta_1 = 0
    else:
        delta_1 = np.arccos(cos_delta_1)
    H2xy_ij = (math.sqrt(k1x * k1y)/(2*math.pi) *
                  (math.sin(delta_1) + (math.pi - delta_1)*cos_delta_1)
                  )
    gp_2_matrix_ij = sigmab22 + c2 * sigmaw22 * H2xy_ij
    
    return gp_2_matrix_ij

#%% Laplacian

def compute_laplace_ij(data1, data2, d_in, i,j, a_cst, b_cst):
    laplace_matrix_ij = a_cst * math.exp(-1*math.sqrt(np.sum(np.square(data1[i][:] - data2[j][:])))/b_cst)
    
    return laplace_matrix_ij

#%% NTKA1

def compute_ntka1_ij(data1, data2, d_in, i,j, c_sigma):
    # Terms for NTKA (1 layer)
    a_k0x = np.sum(np.square(data1[i][:]))
    a_k0y = np.sum(np.square(data2[j][:]))
    a_k0xy = np.sum(data1[i][:] * data2[j][:])
    a_cos_delta_0 = a_k0xy/(a_k0x*a_k0y)
    if(abs(a_cos_delta_0) > 1):
        a_delta_0 = 0
    else:
        a_delta_0 = np.arccos(a_cos_delta_0)
    a_H1xy_ij = (math.sqrt(a_k0x*a_k0y)/(2*math.pi) *
                  (math.sin(a_delta_0)+(math.pi - a_delta_0)*a_cos_delta_0))
    a_H1xy_d_ij = (math.pi - a_delta_0)/(2*math.pi)
    
    # NTKA (1 layer)
    ntka1_matrix_ij = (c_sigma * np.sum(data1[i][:] * data2[j][:]) * a_H1xy_d_ij + 
                    c_sigma * a_H1xy_ij)
    
    return ntka1_matrix_ij


#%% NTKA2

def compute_ntka2_ij(data1, data2, d_in, i,j, c_sigma):
    # Terms for NTKA (1 layer)
    a_k0x = np.sum(np.square(data1[i][:]))
    a_k0y = np.sum(np.square(data2[j][:]))
    a_k0xy = np.sum(data1[i][:] * data2[j][:])
    a_cos_delta_0 = a_k0xy/(a_k0x*a_k0y)
    if(abs(a_cos_delta_0) > 1):
        a_delta_0 = 0
    else:
        a_delta_0 = np.arccos(a_cos_delta_0)
    a_H1xy_ij = (math.sqrt(a_k0x*a_k0y)/(2*math.pi) *
                  (math.sin(a_delta_0)+(math.pi - a_delta_0)*a_cos_delta_0))
    a_H1xy_d_ij = (math.pi - a_delta_0)/(2*math.pi)
    
    # Terms for NTKA (2 layers)
    a_k1x = c_sigma * math.sqrt(a_k0x * a_k0x)/(2*math.pi) * math.pi
    a_k1y = c_sigma * math.sqrt(a_k0y * a_k0y)/(2*math.pi) * math.pi
    a_k1xy = c_sigma * math.sqrt(a_k0x * a_k0y)/(2*math.pi) * (math.sin(a_delta_0) + (math.pi - a_delta_0)*a_cos_delta_0)
    a_cos_delta_1 = a_k1xy/(a_k1x*a_k1y)
    if(abs(a_cos_delta_1) > 1):
        a_delta_1 = 0
    else:
        a_delta_1 = np.arccos(a_cos_delta_1)
    a_H2xy_ij = (
                    math.sqrt(a_k1x * a_k1y)/(2*math.pi) *
                  (math.sin(a_delta_1) + (math.pi - a_delta_1)*a_cos_delta_1)
                  )
    a_H2xy_d_ij = (math.pi - a_delta_1)/(2*math.pi)
    
    # NTKA (2 layers)
    ntka2_matrix_ij = (c_sigma**2 * np.sum(data1[i][:] * data2[j][:]) * a_H1xy_d_ij * a_H2xy_d_ij + 
                    c_sigma**2 * a_H1xy_ij * a_H2xy_d_ij + 
                    c_sigma * a_H2xy_ij
                    )
    
    return ntka2_matrix_ij

#%% NTKB1

def compute_ntkb1_ij(data1, data2, d_in, i,j, sigmab02, sigmaw02, c1, sigmaw12):
    # Terms for NTKB (1 layer)
    k0x = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data1[i][:]))
    k0y = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data2[j][:]))
    k0xy = (sigmab02 +
            sigmaw02/(d_in) *
            np.sum(data1[i][:] * data2[j][:]))
    cos_delta_0 = k0xy/(k0x*k0y)
    if(abs(cos_delta_0) > 1):
        delta_0 = 0
    else:
        delta_0 = np.arccos(cos_delta_0)
    H1xy_ij = (math.sqrt(k0x*k0y)/(2*math.pi) *
                  (math.sin(delta_0)+(math.pi - delta_0)*cos_delta_0))
    H1xy_d_ij = (math.pi - delta_0)/(2*math.pi)
    
    # NTKB (1 layer)
    ntkb1_matrix_ij = (1 +
                    c1 * H1xy_ij +
                    c1 * sigmaw12 * H1xy_d_ij +
                    c1 * sigmaw12 * H1xy_d_ij * np.sum(data1[i][:] * data2[j][:]))
    
    return ntkb1_matrix_ij

#%% NTKB2

def compute_ntkb2_ij(data1, data2, d_in, i,j, sigmab02, sigmaw02, c1, sigmaw12,sigmab12, sigmaw22, c2, ):
    # Terms for NTKB (1 layer)
    k0x = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data1[i][:]))
    k0y = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data2[j][:]))
    k0xy = (sigmab02 +
            sigmaw02/(d_in) *
            np.sum(data1[i][:] * data2[j][:]))
    cos_delta_0 = k0xy/(k0x*k0y)
    if(abs(cos_delta_0) > 1):
        delta_0 = 0
    else:
        delta_0 = np.arccos(cos_delta_0)
    H1xy_ij = (math.sqrt(k0x*k0y)/(2*math.pi) *
                  (math.sin(delta_0)+(math.pi - delta_0)*cos_delta_0))
    H1xy_d_ij = (math.pi - delta_0)/(2*math.pi)
    # Terms for NTKB (2 layers)
    k1x = sigmab12 + c1 * sigmaw12/(2*math.pi) * k0x * math.pi
    k1y = sigmab12 + c1 * sigmaw12/(2*math.pi) * k0y * math.pi
    k1xy = (sigmab12 + 
            c1 * sigmaw12/(2*math.pi) *
            math.sqrt(k0x * k0y) *
            (math.sin(delta_0) + (math.pi - delta_0)*cos_delta_0)
            )
    cos_delta_1 = k1xy/(k1x*k1y)
    if(abs(cos_delta_1) > 1):
        delta_1 = 0
    else:
        delta_1 = np.arccos(cos_delta_1)
    H2xy_ij = (math.sqrt(k1x * k1y)/(2*math.pi) *
                  (math.sin(delta_1) + (math.pi - delta_1)*cos_delta_1)
                  )
    H2xy_d_ij = (math.pi - delta_1)/(2*math.pi)
    # NTKB (2 layers)
    ntkb2_matrix_ij = (1 + 
                    c2 * sigmaw22 * H2xy_d_ij + 
                    c1 * c2 * sigmaw22 * sigmaw12 * H2xy_d_ij * H1xy_d_ij + 
                    c1 * c2 * sigmaw12 * sigmaw22 * np.sum(data1[i][:] * data2[j][:]) * H2xy_d_ij * H1xy_d_ij + 
                    c1 * c2 * sigmaw22 * H2xy_d_ij * H1xy_ij + 
                    c2 * H2xy_ij)
    
    return ntkb2_matrix_ij

#%% NTKJ1 

def compute_ntkj1_ij(data1, data2, d_in, i,j, sigmab02, sigmaw02, c1, sigmaw12):
    # Terms for NTKB (1 layer)
    k0x = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data1[i][:]))
    k0y = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data2[j][:]))
    k0xy = (sigmab02 +
            sigmaw02/(d_in) *
            np.sum(data1[i][:] * data2[j][:]))
    cos_delta_0 = k0xy/(k0x*k0y)
    if(abs(cos_delta_0) > 1):
        delta_0 = 0
    else:
        delta_0 = np.arccos(cos_delta_0)
    H1xy_ij = (math.sqrt(k0x*k0y)/(2*math.pi) *
                  (math.sin(delta_0)+(math.pi - delta_0)*cos_delta_0))
    H1xy_d_ij = (math.pi - delta_0)/(2*math.pi)
    # NTKJ (1 layer)
    ntkj1_matrix_ij = k0xy * H1xy_d_ij + H1xy_ij + 1
    
    return ntkj1_matrix_ij

#%% NTKJ2

def compute_ntkj2_ij(data1, data2, d_in, i,j, sigmab02, sigmab12, sigmaw02, c1, sigmaw12):
    # Terms for NTKB (1 layer)
    k0x = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data1[i][:]))
    k0y = sigmab02 + sigmaw02/(d_in) * np.sum(np.square(data2[j][:]))
    k0xy = (sigmab02 +
            sigmaw02/(d_in) *
            np.sum(data1[i][:] * data2[j][:]))
    cos_delta_0 = k0xy/(k0x*k0y)
    if(abs(cos_delta_0) > 1):
        delta_0 = 0
    else:
        delta_0 = np.arccos(cos_delta_0)
    H1xy_ij = (math.sqrt(k0x*k0y)/(2*math.pi) *
                  (math.sin(delta_0)+(math.pi - delta_0)*cos_delta_0))
    H1xy_d_ij = (math.pi - delta_0)/(2*math.pi)
    # Terms for NTKB (2 layers)
    k1x = sigmab12 + c1 * sigmaw12/(2*math.pi) * k0x * math.pi
    k1y = sigmab12 + c1 * sigmaw12/(2*math.pi) * k0y * math.pi
    k1xy = (sigmab12 + 
            c1 * sigmaw12/(2*math.pi) *
            math.sqrt(k0x * k0y) *
            (math.sin(delta_0) + (math.pi - delta_0)*cos_delta_0)
            )
    cos_delta_1 = k1xy/(k1x*k1y)
    if(abs(cos_delta_1) > 1):
        delta_1 = 0
    else:
        delta_1 = np.arccos(cos_delta_1)
    H2xy_ij = (math.sqrt(k1x * k1y)/(2*math.pi) *
                  (math.sin(delta_1) + (math.pi - delta_1)*cos_delta_1)
                  )
    H2xy_d_ij = (math.pi - delta_1)/(2*math.pi)
    
    # NTKJ (1 layer)
    ntkj1_matrix_ij = k0xy * H1xy_d_ij + H1xy_ij + 1
    
    # NTKJ (2 layers)
    ntkj2_matrix_ij = ntkj1_matrix_ij * H2xy_d_ij + H2xy_ij + 1
    
    return ntkj2_matrix_ij
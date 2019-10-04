#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic code for cosine series fitting."""

import numpy as np
import pandas as pd
import scipy.stats as stats

def load_data():

    fname = 'data_360_1_H.csv'
    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding="latin-1")
    # xdata: dihedral angles
    xdata = data.values[:,0]
    # ydata: rel. PES in meV
    ydata = data.values[:,5]

    return (xdata, ydata)

def linear_regression(x, t, basis, reg_lambda=0, degree=0):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    
    phi = design_matrix(x, basis, degree)
    #print(phi.size)
    
    # TO DO:: Compute coefficients using phi matrix

    if reg_lambda == 0:
        phi_pseudo = np.linalg.pinv(phi)   # pseudo-inverse matrix
        w = np.dot(phi_pseudo, t)          # product of two matrices

    else:
        phi_t = np.transpose(phi)
        fo = reg_lambda * np.eye(degree+1) + np.dot(phi_t, phi)
        fo_inv = np.linalg.inv(fo)
        po = np.dot(fo_inv, phi_t)
        w = np.dot(po, t)
        
    # TO DO:: Measure root mean squared error on training data.
    
    w_t = np.transpose(w)   # transpose of matrix w
    N = len(x)              # the number of training data
    sum_err = 0
    
    for i in range(0, N): 
        sum_err = sum_err + np.power(np.dot(w_t, np.transpose(phi[i,:])) - t[i], 2) 
    
    train_err = np.sqrt(sum_err/N)  # RMS error on training dataset

    return (w, train_err)

def design_matrix(x, basis, degree=0):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions
    if basis == 'polynomial':
        N = len(x)        # the number of training data
        s = (N, 1)   
        phi = np.ones(s)  # as the first column of matrix phi is x^0 (just 1) so np.ones used
        
        for i in range(1 , degree+1):  
            phi = np.column_stack((phi, np.power(x, i)))  # concatenating column-wise
    
    elif basis == 'cosine':
        N = len(x)        # the number of training data
        s = (N, 1)
        phi = np.ones(s)
        
        for i in range(1 , degree+1):  
            phi = np.column_stack((phi, np.cos(i*x*np.pi/180)))  # concatenating column-wise
            
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi

def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
    # TO DO:: Compute t_est and err 
    phi = design_matrix(x, basis, degree)
    w_t = np.transpose(w)     # transpose of matrix w
    N = len(x)                # the number of test data
    sum_err = 0
    t_est = np.zeros(N)       # values of regression on test data   
    
    for i in range(0, N):
        t_est[i] = np.dot(w_t, np.transpose(phi[i,:]))
        sum_err = sum_err + np.power(t_est[i] - t[i], 2) 
    
    err = np.sqrt(sum_err/N)  # RMS error on testing dataset
    return (t_est, err)

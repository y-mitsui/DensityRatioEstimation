# -*- coding:utf-8 -*-
# LSIF
from scipy.stats import binom
import pandas as pd
import numpy as np
import sys
from scipy.stats import norm
from numpy.random import *
import theano
import theano.tensor as T
import math
import matplotlib.pyplot as plt

class LSIF:
    """
        r(sample) = P_x(sample) / P_y(sample)
    """
    def __init__(self, band_width, regulation, n_kernel_fold=20):
        self.band_width = band_width
        self.regulation = regulation
        self.n_kernel_fold = n_kernel_fold
    
    def kernel(self, X):
        result = []
        for X2 in self.sample_kernel_fold:
            diff_vec = X - X2
            result.append(np.exp(-np.dot(diff_vec, diff_vec) / (2*self.band_width**2)))
        return np.array(result)
        
    def kernel2(self, x, x2, band_width):
        h = band_width
        return np.exp(-(x-x2)**2/(2*h**2))
        
    def estGHat(self, sample_hospital):
        g_hat = np.matrix(np.zeros((self.n_kernel_fold, self.n_kernel_fold)),dtype=float)
        for sample in sample_hospital:
            basis_result = np.matrix(self.kernel(sample))
            g_hat += basis_result.T * basis_result
        g_hat /= sample_hospital.shape[0]
        return g_hat

    def estHHat(self, sample_entry):
        h_hat = np.matrix(np.zeros(self.n_kernel_fold),dtype=float)
        for sample in sample_entry:
            basis_result = np.matrix(self.kernel(sample))
            h_hat += basis_result
        h_hat /= sample_entry.shape[0]
        return h_hat
        
    def fit(self, sample_X, sample_Y):
        self.sample_kernel_fold = sample_X[:self.n_kernel_fold]
        g_hat = self.estGHat(sample_Y)
        h_hat = self.estHHat(sample_X)
        self.thetas = np.linalg.inv(g_hat + self.regulation * np.matrix(np.identity(self.n_kernel_fold))) * h_hat.T
        self.thetas = np.maximum(self.thetas, 0)
        
    def predict(self, sample):
        result = []
        for x2 in sample:
            r = self.thetas.T * np.matrix(self.kernel(x2)).T
            result.append(r[0,0])
        return np.array(result)
    
def model(x,sampleEntry,thetas, band_width):

    sample = []
    for x2 in x:
        r = thetas.T * np.matrix(kernel(x2, sampleEntry, band_width)).T
        sample.append(r[0,0])
    return np.array(sample)
    
def kernel(x, x2, band_width):
    h = band_width
    return np.exp(-(x-x2)**2/(2*h**2))
    
def normNg(theta, g_hat, h_hat, stop):
    return 0.5 * theta.T * g_hat * theta - theta.T * h_hat.T

def estGHat(sample_hospital, sample_entry_train, band_width):
    g_hat = np.matrix(np.zeros((num_theta,num_theta)),dtype=float)
    for sample in sample_hospital:
        basis_result = np.matrix(kernel(sample, sample_entry_train, band_width))
        g_hat += basis_result.T * basis_result
    g_hat /= sample_hospital.shape[0]
    return g_hat

def estHHat(sample_entry, sample_entry_train, band_width):
    h_hat = np.matrix(np.zeros(num_theta),dtype=float)
    for sample in sample_entry:
        basis_result = np.matrix(kernel(sample,sample_entry_train, band_width))
        h_hat += basis_result
    h_hat /= sample_entry.shape[0]
    return h_hat
    
if __name__ == "__main__":

    np.random.seed(123)

    n_sample = 5000
    scale=3.

    sample_X = randn(n_sample, 1) * scale + 1.5
    sample_Y = randn(n_sample, 1) * scale + 1
    lsif = LSIF(3.3, 0.25, sample_X.shape[0])
    lsif.fit(sample_X, sample_Y)
    prob_true = norm.pdf(sample_Y, loc=1.5, scale=scale) / norm.pdf(sample_Y, loc=1., scale=scale)
    prob_est = lsif.predict(sample_Y)
    plt.scatter(prob_est, prob_true)
    plt.show()
    sys.exit(1)

    n_sample = 1000
    stop = 2.0

    sample_entry = randn(n_sample) * scale + 1.5
    sample_hospital = randn(n_sample) * scale + 1
    num_group = 20

    n_test = int(sample_entry.shape[0] * 0.3)
    #num_theta = n_sample - n_test
    sample_entry_train = sample_entry[:-n_test]
    sample_entry_test = sample_entry[n_test:]
    num_theta = sample_entry_train.shape[0]
    sample_hospital_train = sample_hospital[:-n_test]
    sample_hospital_test = sample_hospital[n_test:]

    min_error = float('inf')

    for band_width in np.linspace(2.5, 5, 10):
        for regration in  np.linspace(5e-2, 2, 10):
            print band_width, regration
            g_hat = estGHat(sample_hospital_train, sample_entry_train[:num_theta], band_width)
            h_hat = estHHat(sample_entry_train, sample_entry_train[:num_theta], band_width)

            thetas = np.linalg.inv(g_hat + regration * np.matrix(np.identity(num_theta))) * h_hat.T
            thetas = np.maximum(thetas, 0)
            print thetas[:10]
            
            g_hat = estGHat(sample_hospital_test, sample_entry_train[:num_theta], band_width)
            h_hat = estHHat(sample_entry_test, sample_entry_train[:num_theta], band_width)
            cur_error = normNg(thetas, g_hat, h_hat, regration)
            print "current error ", cur_error, band_width, regration
            if min_error > cur_error:
                min_error = cur_error
                min_params = (thetas, sample_entry_train[:num_theta], band_width)
                print "current best ", min_error, band_width, regration
      
    """
    sample_entry = randn(n_sample) * scale + 3
    sample_hospital = randn(n_sample) * scale + 1
    num_theta = n_sample
    band_width = 2
    regration = 2
    g_hat = estGHat(sample_hospital, sample_entry[:num_theta], band_width)
    h_hat = estHHat(sample_entry, sample_entry[:num_theta], band_width)
    thetas = np.linalg.inv(g_hat + regration * np.matrix(np.identity(num_theta))) * h_hat.T
    min_params = (thetas, sample_entry[:num_theta], band_width)
    """

    prob = norm.pdf(sample_hospital, loc=1.5, scale=scale) / norm.pdf(sample_hospital, loc=1., scale=scale)
    prob2 = model(sample_hospital, min_params[1], min_params[0], min_params[2])
    plt.scatter(prob2, prob)
    plt.show()
    sys.exit(1)
    for row1,row2 in zip(prob,prob2):
        
        print "True Value:%f Estimate Value:%f"%(row1,row2)

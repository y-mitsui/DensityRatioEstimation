# -*- coding:utf-8 -*-
# uLSIF
from scipy.stats import binom
import pandas as pd
import numpy as np
import sys
from scipy.stats import norm
from numpy.random import *
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
    
    def _score(self, theta, g_hat, h_hat):
        return 0.5 * theta.T * g_hat * theta - theta.T * h_hat.T

    def score(self, sample_X, sample_Y):
        g_hat = self.estGHat(sample_X)
        h_hat = self.estHHat(sample_Y)
        return self._score(self.thetas, g_hat, h_hat) 
        
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

    
if __name__ == "__main__":

    np.random.seed(123)

    n_sample = 500
    scale=3.

    sample_X = randn(n_sample, 1) * scale + 1.5
    sample_Y = randn(n_sample, 1) * scale + 1
    
    n_test = int(sample_X.shape[0] * 0.3)
    sample_X_train = sample_X[:-n_test]
    sample_X_test = sample_X[n_test:]
    num_theta = sample_X.shape[0]
    sample_Y_train = sample_Y[:-n_test]
    sample_Y_test = sample_Y[n_test:]
    
    min_error = float('inf')
    for band_width in np.linspace(2.5, 5, 0):
        for regration in  np.linspace(5e-2, 2, 10):
            lsif = LSIF(band_width, regration, sample_X_train.shape[0])
            lsif.fit(sample_X_train, sample_Y_train)
            score = lsif.score(sample_X_test, sample_Y_test)
            print band_width, regration, score
            if min_error > score:
                min_error = score
                min_params = (band_width, regration)
                print "current best ", band_width, regration, min_error
            
    lsif = LSIF(3.61111111111, 0.48, sample_X.shape[0])
    lsif.fit(sample_X, sample_Y)
    prob_true = norm.pdf(sample_X, loc=1.5, scale=scale) / norm.pdf(sample_X, loc=1., scale=scale)
    prob_est = lsif.predict(sample_X)
    plt.scatter(prob_est, prob_true)
    plt.show()
    

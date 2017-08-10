# -*- coding:utf-8 -*-
# uLSIF
import numpy as np

class LSIF:
    """
        Density radio estimation of  uLSIF (Sugiyama) using RBF kernel
        r(sample) = P_x(sample) / P_y(sample)
        
        example:
            LSIF(0.3, 0.1).fit(sample_molecule, sample_denominator).predict(sample_new)
    """
    
    def __init__(self, band_width, regulation):
        """
        @param band_width: parameter of RBF kernel
        @param regulation: regulation to prevent over fitting
        """
        self.band_width = band_width
        self.regulation = regulation
    
    def kernel(self, X):
        result = []
        for X2 in self.sample_kernel_fold:
            diff_vec = X - X2
            result.append(np.exp(-np.dot(diff_vec, diff_vec) / (2*self.band_width**2)))
        return np.array(result)
        
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
        g_hat = self.estGHat(sample_Y)
        h_hat = self.estHHat(sample_X)
        return self._score(self.thetas, g_hat, h_hat) 
        
    def fit(self, sample_X, sample_Y):
        self.n_kernel_fold = sample_X.shape[0]
        self.n_kernel_fold = 100
        self.sample_kernel_fold = sample_X[:self.n_kernel_fold]
        g_hat = self.estGHat(sample_Y)
        h_hat = self.estHHat(sample_X)
        self.thetas = np.linalg.inv(g_hat + self.regulation * np.matrix(np.identity(self.n_kernel_fold))) * h_hat.T
        self.thetas = np.maximum(self.thetas, 0)
        return self
        
    def predict(self, sample):
        result = []
        for x2 in sample:
            r = self.thetas.T * np.matrix(self.kernel(x2)).T
            result.append(r[0,0])
        return np.array(result)

    

    

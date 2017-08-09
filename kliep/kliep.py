# -*- coding:utf-8 -*-
# Kullback-Leibler Importance Estimation Procedure
import numpy as np
import sys
from scipy.stats import norm
from numpy.random import *
import math
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from scipy.optimize import minimize
from cuda_kliep_wrap import cudaKliepWrap
import time

class CudaKLIEP:
    """
        r(sample) = P_x(sample) / P_y(sample)
    """
    
    def __init__(self, band_width, n_kernel_fold=10, learning_rate=1e-2, n_iter=10000, load_file=None):
        if load_file != None:
            self.load(load_file)
            self.cuda_kliep = cudaKliepWrap(self.n_kernel_fold, self.band_width, self.n_iter, self.learning_rate, self.theta)
        else:
            theta = np.random.rand(n_kernel_fold)
            self.cuda_kliep = cudaKliepWrap(n_kernel_fold, band_width, n_iter, learning_rate, theta)
            
            self.band_width = band_width
            self.n_kernel_fold = n_kernel_fold
            self.learning_rate = learning_rate
            self.n_iter = n_iter
            self.kernel_param = -1 / (2 * self.band_width ** 2)
        
    def kernel(self, X):
        result = []
        for X2 in self.sample_kernel_fold:
            diff_vec = X - X2
            result.append(np.exp(-np.dot(diff_vec, diff_vec) / (2*self.band_width**2)))
        return np.array(result)
    
    def _score(self, theta, sample_entry, sample_hospital):
        r_hospital = np.matrix(np.zeros(1))
        for each_sample in sample_hospital:
            r_hospital += theta * np.matrix(self.kernel(each_sample)).T
        r_hospital /= sample_hospital.shape[0]
        
        r_entry = np.matrix(np.zeros(1))
        for each_sample in sample_entry:
            r_entry += np.dot(theta, self.kernel(each_sample))
        r_entry /= sample_entry.shape[0]
        
        return r_hospital[0, 0] - r_entry[0, 0]
        
    def score(self, sample_entry, sample_hospital):
        return self._score(self.theta, sample_entry, sample_hospital)
        
    def load(self, file_prefix):
        self.theta = np.load(file_prefix + "_theta.npy")[0]
        self.sample_kernel_fold = np.load(file_prefix + "_sample_kernel_fold.npy")
        self.n_kernel_fold, self.band_width, self.n_iter, self.learning_rate = np.load(file_prefix + "_setting.npy")
        self.n_kernel_fold, self.n_iter = int(self.n_kernel_fold), int(self.n_iter)
        
    def save(self, file_prefix):
        np.save(file_prefix + "_theta", self.theta)
        np.save(file_prefix + "_sample_kernel_fold", self.sample_kernel_fold)
        np.save(file_prefix + "_setting", (self.n_kernel_fold, self.band_width, self.n_iter, self.learning_rate))
        
    def differentialTargetFunction(self, thetas, sampleEntry, sampleHospital):
        
        sum_basis_result2 = np.matrix(np.zeros(thetas.shape[1]))
        for basis_result in self.basis_results:
            sum_basis_result2 += basis_result / (thetas * basis_result.T)
        sum_basis_result2 /= sampleEntry.shape[0]
        
        return self.sum_basis_result  - sum_basis_result2
        
    def fit(self, sample_X, sample_Y):
        sample_X = np.array(sample_X)
        sample_Y = np.array(sample_Y)
        
        self.theta = self.cuda_kliep.fit(sample_X, sample_Y)
        self.theta = np.matrix(self.theta)
        
        self.sample_kernel_fold = sample_X[:self.n_kernel_fold]
        
        """
        self.sample_kernel_fold = sample_X[:self.n_kernel_fold]
        self.theta = np.random.rand(self.n_kernel_fold)
        theta_previous = self.theta
        error_previous = float('inf')
        
        basis_result = [self.kernel(sample) for sample in sample_Y]
        self.sum_basis_result = np.average(basis_result, axis=0)
        self.basis_results = [self.kernel(sample) for sample in sample_X]
        
        self.theta = cudaKliepOptWrap(self.sum_basis_result, np.array(self.basis_results), self.n_iter, self.learning_rate, self.theta)
        self.theta = np.matrix(self.theta)
        """
        
    def predict(self, sample):
        return self.cuda_kliep.predict(sample, self.sample_kernel_fold)
        """
        result = []
        for x2 in sample:
            r = np.dot(self.theta, self.kernel(x2))
            result.append(r[0, 0])
        """ 

"""
class KLIEP:
    
        r(sample) = P_x(sample) / P_y(sample)
    
    
    def __init__(self, band_width, n_kernel_fold=10, learning_rate=1e-2, n_iter=10000, tol=1e-5):
        self.band_width = band_width
        self.n_kernel_fold = n_kernel_fold
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.kernel_param = -1 / (2 * self.band_width ** 2)
        
    def kernel(self, X):
        result = []
        for X2 in self.sample_kernel_fold:
            diff_vec = X - X2
            result.append(np.exp(-np.dot(diff_vec, diff_vec) / (2*self.band_width**2)))
        return np.array(result)
    
    def _score(self, theta, sample_entry, sample_hospital):
        r_hospital = np.matrix(np.zeros(1))
        for each_sample in sample_hospital:
            r_hospital += theta * np.matrix(self.kernel(each_sample)).T
        r_hospital /= sample_hospital.shape[0]
        
        r_entry = np.matrix(np.zeros(1))
        for each_sample in sample_entry:
            r_entry += np.dot(theta, self.kernel(each_sample))
        r_entry /= sample_entry.shape[0]
        
        return r_hospital[0, 0] - r_entry[0, 0]
        
    def score(self, sample_entry, sample_hospital):
        return self._score(self.theta, sample_entry, sample_hospital)
        
    def differentialTargetFunction(self, thetas, sampleEntry, sampleHospital):
        
        sum_basis_result2 = np.matrix(np.zeros(thetas.shape[1]))
        for basis_result in self.basis_results:
            sum_basis_result2 += basis_result / (thetas * basis_result.T)
        sum_basis_result2 /= sampleEntry.shape[0]
        
        return self.sum_basis_result  - sum_basis_result2
        
    def fit(self, sample_X, sample_Y):
        sample_X = np.array(sample_X)
        sample_Y = np.array(sample_Y)
        
        self.sample_kernel_fold = sample_X[:self.n_kernel_fold]
        self.theta = np.random.rand(self.n_kernel_fold)
        theta_previous = self.theta
        error_previous = float('inf')
        
        basis_result = [self.kernel(sample) for sample in sample_Y]
        self.sum_basis_result = np.average(basis_result, axis=0)
        self.basis_results = [self.kernel(sample) for sample in sample_X]
        
        self.theta = cudaKliepOptWrap(self.sum_basis_result, np.array(self.basis_results), self.n_iter, self.learning_rate, self.theta)
        self.theta = np.matrix(self.theta)
        print "self.theta.shape", self.theta.shape
        
        
        target_function = DiffTargetFunction(np.array(self.basis_results))
        
        t1 = time.time()
        for i in range(self.n_iter):
            self.theta -= self.learning_rate * (self.sum_basis_result - target_function.calc(self.theta))
            
            if i % 50 == 0:
                #error_cur = self.score(sample_X, sample_Y)
                #print("%d / %d (%f) %fsec"%(i, self.n_iter, error_previous - error_cur, time.time() - t1))
                print("%d / %d %fsec"%(i, self.n_iter, time.time() - t1))
                t1 = time.time()
                print(self.theta[:20])
                #error_previous = error_cur
                
        self.theta = np.matrix(self.theta)
        
    def predict(self, sample):
        result = []
        for x2 in sample:
            r = np.dot(self.theta, self.kernel(x2))
            result.append(r[0, 0])
            
        return np.array(result)
"""

def optimize(num_theta,sampleEntry,sampleHospital):
    theta = np.matrix( randn(num_theta) * 0.5 )
    delta = 0.02
    error_previous = float('inf')
    
    for i in range(1500):
        theta -= delta * differentialTargetFunction(theta,sampleEntry,sampleHospital)
        if i % 100 == 0:
            error_cur = errorFunction(theta, sampleEntry[:theta.shape[1]], sampleEntry, sampleHospital, band_width)
            if error_previous < error_cur:
                raise Exception('Failed learning')
                
            if error_previous - error_cur < 1e-5:
                break
            
            print("--------- %d ---------"%(i))
            print(error_previous - error_cur)
            print(theta)
            error_previous = error_cur
            
    return theta

def errorFunction(theta, kernel_sample, sample_entry, sample_hospital, band_width):
    r_hospital = np.matrix(np.zeros(1))
    for each_sample in sample_hospital:
        r_hospital += theta * np.matrix(kernel(each_sample, kernel_sample, band_width)).T
    r_hospital /= sample_hospital.shape[0]
    
    r_entry = np.matrix(np.zeros(1))
    for each_sample in sample_entry:
        r_entry += np.dot(theta, kernel(each_sample, kernel_sample, band_width))
    r_entry /= sample_entry.shape[0]
    
    return r_hospital[0, 0] - r_entry[0, 0]
    
def kernel2(x, x2, band_width):
    h = band_width
    return np.exp(-np.sqrt((x-x2)**2)/(2*h**2))
    
def kernel(x, x2, band_width):
    h = band_width
    result = []
    for vec in x2:
        diff_vec = x - vec
        result.append(np.exp(-np.sqrt(np.dot(diff_vec, diff_vec)) / (2*h**2)))
    return np.array(result)

def model(x,sampleEntry,thetas):

    sample = []
    for x2 in x:
        r = thetas * np.matrix(kernel(x2,sampleEntry[:thetas.shape[1]], band_width)).T
        sample.append(r[0,0])
    return np.array(sample)

def differentialTargetFunction(thetas,sampleEntry,sampleHospital):
    basis_result = [kernel(sample,sampleEntry[:thetas.shape[1]], band_width) for sample in sampleHospital]
    sum_basis_result = np.average(basis_result,axis=0)
    
    sum_basis_result2 = np.matrix(np.zeros(thetas.shape[1]))
    for sample in sampleEntry:
        basis_result = np.matrix(kernel(sample,sampleEntry[:thetas.shape[1]], band_width))
        sum_basis_result2 += basis_result / (thetas * basis_result.T)
    sum_basis_result2 /= sampleEntry.shape[0]
    
    return sum_basis_result - sum_basis_result2

if __name__ == "__main__":
    np.random.seed(123)
    n_sample_entry = 30
    n_sample_hospital = 50
    n_kernel_fold = n_sample_entry
    sample_entry = randn(n_sample_entry, 1) * 2 + 2
    sample_hospital = randn(n_sample_hospital, 1) * 2 + 1
    n_test = int(sample_entry.shape[0] * 0.3)

    n_fold = 20

    min_error = float('inf')
    for band_width in np.linspace(1, 10, 0):
        print(band_width)
        kliep = CudaKLIEP(band_width, learning_rate=5e-2, n_iter=20000, n_kernel_fold=n_kernel_fold)

        mean_score = 0.
        for i, ((hospital_train, hospital_test), (entry_train, entry_test)) in enumerate(zip(KFold(sample_hospital.shape[0], n_folds=n_fold), KFold(sample_entry.shape[0], n_folds=n_fold))):
            if i > 3:
                break
                
            sample_hospital_train, sample_hospital_test = sample_hospital[hospital_train], sample_hospital[hospital_test]
            sample_entry_train, sample_entry_test = sample_entry[entry_train], sample_entry[entry_test]
            kliep.fit(sample_entry_train, sample_hospital_train)
            cur_score = kliep.score(sample_entry_test, sample_hospital_test)
            print("cur_score:", cur_score)
            mean_score += cur_score
        mean_score /= n_fold
        
        print("band_width:", band_width, "mean_score:", mean_score)
        if min_error > mean_score:
            min_error = mean_score
            min_param = band_width
        
    band_width = 1.5
    print("best band_width", band_width)
    kliep = CudaKLIEP(band_width, learning_rate=5e-2, n_iter=5000, n_kernel_fold=n_kernel_fold)
    kliep.fit(sample_entry, sample_hospital)
    kliep.save("test_kliep")
    kliep = CudaKLIEP(band_width, load_file="test_kliep")
    est_ratio = 1. / kliep.predict(sample_entry)
    true_ratio = norm.pdf(sample_entry,loc=1,scale=2) / norm.pdf(sample_entry,loc=2,scale=2)
    plt.scatter(est_ratio, true_ratio)
    plt.show()
    


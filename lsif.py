# -*- coding:utf-8 -*-
# LSIF
from scipy.stats import binom
import pandas as pd
import numpy
import sys
from scipy.stats import norm
from numpy.random import *
import theano
import theano.tensor as T
import math

import matplotlib.pyplot as plt

def model(x,sampleEntry,thetas):

    sample = []
    for x2 in x:
        r = thetas.T * numpy.matrix(kernel(x2,sampleEntry[:thetas.shape[0]])).T
        sample.append(r[0,0])
    return numpy.array(sample)
    
def kernel(x,x2):
    h=2
    return numpy.exp(-(x-x2)**2/(2*h**2))
def normNg(theta,g_hat,h_hat,stop):
    return 0.5 * theta.T * g_hat * theta - theta.T * h_hat.T + stop / 2 * ( theta.T * theta )

numpy.random.seed(123)

num_theta = 500
stop = 3.0
scale=3.
sample_entry = randn(500) * scale + 3
sample_hospital = randn(500) * scale + 1
num_group = 20
n_fold_entry = 
n_fold_hospital =

sample_entries = [sample_entry[i:i+n_fold] for i in range(num_group)]
sample_hospitals = [sample_hospital[i:i+n_fold] for i in range(num_group)]


g_hat = numpy.matrix(numpy.zeros((num_theta,num_theta)),dtype=float)
for sample in sample_hospital:
    basis_result = numpy.matrix(kernel(sample,sample_entry[:num_theta]))
    g_hat += basis_result.T * basis_result
g_hat /= sample_hospital.shape[0]

h_hat = numpy.matrix(numpy.zeros(num_theta),dtype=float)
for sample in sample_entry:
    basis_result = numpy.matrix(kernel(sample,sample_entry[:num_theta]))
    h_hat += basis_result
h_hat /= sample_entry.shape[0]

thetas = numpy.linalg.inv(g_hat + stop * numpy.matrix(numpy.identity(num_theta))) * h_hat.T
print normNg(thetas,g_hat,h_hat,stop)
#sys.exit(1)
prob = norm.pdf(sample_hospital,loc=2,scale=scale) / norm.pdf(sample_hospital,loc=1,scale=scale)
prob2 = model(sample_hospital,sample_entry,thetas)
plt.scatter(prob2, prob)
plt.show()
sys.exit(1)
for row1,row2 in zip(prob,prob2):
    
    print "True Value:%f Estimate Value:%f"%(row1,row2)

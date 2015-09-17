# -*- coding:utf-8 -*-
# Kullback-Leibler Importance Estimation Procedure
from scipy.stats import binom
import pandas as pd
import numpy
import sys
from scipy.stats import norm
from numpy.random import *

def optimize(num_theta,sampleEntry,sampleHospital):
    theta = numpy.matrix( [1.]*num_theta )
    delta = 5e-3
    for i in range(10000):
        theta -= delta*differentialTargetFunction(theta,sampleEntry,sampleHospital)
        if i % 100 == 0:
            print "--------- %d ---------"%(i)
            print theta
    return theta

def kernel(x,x2):
    h=0.5
    return numpy.exp(-numpy.sqrt((x-x2)**2)/(2*h**2))

def model(x,sampleEntry,thetas):

    sample = []
    for x2 in x:
        r = thetas * numpy.matrix(kernel(x2,sampleEntry[:thetas.shape[1]])).T
        sample.append(r[0,0])
    return numpy.array(sample)

def differentialTargetFunction(thetas,sampleEntry,sampleHospital):
    
    basis_result = [kernel(sample,sampleEntry[:thetas.shape[1]]) for sample in sampleHospital]
    sum_basis_result = numpy.average(basis_result,axis=0)
    
    sum_basis_result2 = numpy.matrix(numpy.zeros(thetas.shape[1]))
    for sample in sampleEntry:
        basis_result = numpy.matrix(kernel(sample,sampleEntry[:thetas.shape[1]]))
        sum_basis_result2 += basis_result / (thetas * basis_result.T)
    sum_basis_result2 /= thetas.shape[1]
    
    return sum_basis_result - sum_basis_result2

sample_entry = randn(500) + 12
sample_hospital = randn(500) + 10
thetas = optimize(500,sample_entry,sample_hospital)

prob = norm.pdf(sample_hospital,loc=11) / norm.pdf(sample_hospital,loc=10,scale=1)
prob2 = model(sample_hospital,sample_entry,thetas)
for row1,row2 in zip(prob,prob2):
    print "True Value:%f Estimate Value:%f"%(row1,row2)


# -*- coding:utf-8 -*-
# Kullback-Leibler Importance Estimation Procedure
from scipy.stats import binom
import pandas as pd
import numpy
import sys
from scipy.stats import norm
from numpy.random import *
import theano
import theano.tensor as T
import math
def f(x):
    return norm.pdf(sample_entry,loc=2,scale=2)
def optimize2():
    x = 1.0   
    delta = 100
    for i in range(10000000):
        x -= delta * gPosterior(x,10,10)
        delta *= 0.99999
        if i % 1000 == 0:
            print "--------- %d ---------"%(i)
            print x
            print delta
    return x
    
def optimize(num_theta,sampleEntry,sampleHospital):
    theta = numpy.matrix( randn(num_theta) )
    delta = 0.09
    for i in range(100000):
        theta -= delta * differentialTargetFunction(theta,sampleEntry,sampleHospital)
        if i % 100 == 0:
            print "--------- %d ---------"%(i)
            print theta
    return theta

def kernel(x,x2):
    h=4.
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
    sum_basis_result2 /= sampleEntry.shape[0]
    
    return sum_basis_result - sum_basis_result2

x = T.dscalar('x')
u = T.dscalar('u')
sigma = T.dscalar('sigma')
normalPdfSyntax = -(1./(T.sqrt(2*math.pi*sigma)) * T.exp(-(x-u)**2/(2*sigma)))
posterior = theano.function(inputs=[x,u,sigma], outputs=normalPdfSyntax)
#事後分布の導関数を生成
gPosteriorSyntax = T.grad(cost=normalPdfSyntax, wrt=x)
gPosterior = theano.function(inputs=[x,u,sigma], outputs=gPosteriorSyntax)
optimize2()
sys.exit(1)
sample_entry = randn(200) * 2 + 2
sample_hospital = randn(200) * 2 + 1
thetas = optimize(10,sample_entry,sample_hospital)

prob = norm.pdf(sample_entry,loc=2,scale=2) / norm.pdf(sample_hospital,loc=1,scale=2)
prob2 = model(sample_hospital,sample_entry,thetas)
for row1,row2 in zip(prob,prob2):
    print "True Value:%f Estimate Value:%f"%(row1,row2)


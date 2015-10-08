# -*- coding:utf-8 -*-
import numpy
import random
import sys
import math
import theano
import theano.tensor as T
import time

def softmax(param):
    
    return numpy.exp(param)/numpy.exp(param).sum()
    
def find_reasonable_epsilon(parameter, energy_function, energy_function_delta):
    epsilon = 3e-4
    momentum = numpy.random.normal(0., 1., len(parameter))
    hamilton = energy_function(parameter) - numpy.sum(momentum**2.0) / 2.0
    current_parameter_condinate, momentum = leapfrog(parameter, momentum, epsilon, energy_function_delta)
    differenceHamilton = (energy_function(current_parameter_condinate) - numpy.sum(momentum ** 2.0) / 2.0)  - hamilton
    accept_probability = numpy.exp(differenceHamilton)
    

    a = 2 * int( accept_probability > 0.5) - 1

    while accept_probability ** a > 2 **-a :
        
        epsilon = 2. ** a * epsilon
        current_parameter_condinate, momentum = leapfrog(current_parameter_condinate, momentum, epsilon, energy_function_delta)
        differenceHamilton = (energy_function(current_parameter_condinate) - numpy.sum(momentum ** 2.0) / 2.0) - hamilton
        accept_probability = numpy.exp(differenceHamilton)
        
    print "find_reasonable_epsilon=", epsilon ** a
    return epsilon

def leapfrog(current_parameter_condinate,momentum,step_accuracy,energy_function_delta):
    momentum = momentum + (step_accuracy/2.) * energy_function_delta(current_parameter_condinate)
    current_parameter_condinate = current_parameter_condinate + step_accuracy * momentum
    current_parameter_condinate = numpy.array([max(min(x,20),-20) for x in current_parameter_condinate])
    momentum = momentum + (step_accuracy/2.) * energy_function_delta(current_parameter_condinate)
    
    return current_parameter_condinate, momentum     
    
def hamiltonianMonteCarlo(parameter, energy_function, energy_function_delta,argument=None, iter=1000,bear_in=800,frequency=10, iter_leapfrog=40):
    """
    ハミルトニアンモンテカルロ法によるベイズ推定
    :param  parameter: numpy.ndarray: 初期パラメーター
    :param  energy_function: callable: 事後分布の対数尤度を返す関数
    :param energy_function_delta: callable: 事後分布の導関数
    :param argument: object: energy_function及びenergy_function_deltaに渡す引数
    :param iter: int: MCMCの繰り返し回数
    :param bear_in: int: サンプリング前の準備回数
    :param frequency: int: サンプリング頻度
    :param iter_leapfrog: int: リープフレッグ繰り返し数
    """
    target_accept = 0.8
    t0 = 10
    gamma = 0.05
    average_H = 0.0
    
    current_parameter = parameter
    r = []
    
    step_size  = find_reasonable_epsilon(parameter, energy_function,energy_function_delta)
    u = numpy.log(10 * step_size)
    
    for i in range(iter):
        momentum = numpy.random.randn(len(parameter))
        hamilton = energy_function(current_parameter) - numpy.sum(momentum**2.0) / 2.0
        current_parameter_condinate = current_parameter
        
        for _ in range(iter_leapfrog):
            try:
                current_parameter_condinate, momentum = leapfrog(current_parameter_condinate,momentum,step_size,energy_function_delta)
            except Exception, e:
                print e
                continue
        
        differenceHamilton = ( (energy_function(current_parameter_condinate) - numpy.sum(momentum ** 2.0) / 2.0) ) - hamilton

        accept_probability = min(1.,numpy.exp(differenceHamilton))
        if i < bear_in:
            H_t = target_accept - accept_probability
            w = 1. / ( i + t0 )
            average_H = (1 - w) * average_H + w * H_t
            step_size = min(10.,numpy.exp(u - (numpy.sqrt(i)/gamma)*average_H))
            #step_size = 0.5
            if i % 1000 == 0:
                print "------ %d ------"%(i)
                print "accept_probability:{}".format(accept_probability)
                print "average_H:{}".format(average_H)
                print "step_size:{}".format(step_size)

        if random.random() < numpy.exp(differenceHamilton):
            current_parameter = current_parameter_condinate

        
        if i % 1000 == 0:
            print "====== %d ======"%(i)
            print "{0}".format(softmax(current_parameter))
            print "likelyfood:%f"%(energy_function(current_parameter_condinate))

        if i > bear_in and i % frequency == 0:
            r.append(current_parameter)
        
    return numpy.average(r,axis=0)
    
def bern(p):
    return numpy.random.uniform() < p
        
def callPosterior(parameter):
    return posterior(parameter[:-1],1./parameter.shape[0],parameter[-1])
    
def callGPosterior(parameter):
    diffence_values = gPosterior(parameter[:-1],1./parameter.shape[0],parameter[-1])
    diffence_values = numpy.hstack(diffence_values)
    if any(diffence_values != diffence_values):
        print "------------------ Catch NaN ------------------------"
        print diffence_values
        sys.exit(1)
        raise Exception
    return diffence_values

if __name__ == "__main__":
    x = T.dvector('x')
    u = T.dscalar('u')
    sigma = T.dscalar('sigma')
    sample = numpy.array([ 367,86,20,85,142,530,86,3,1,129,17,47,41,27,17,
                1999,834,574,215,302,84,96,15,2,141,32,7,12])
    n = theano.shared(sample, name='n')
    normalPdfSyntax = - (x.shape[0] / 2.) * T.log( 2. * math.pi * sigma ** 2) - (1./(2*sigma ** 2)) * T.sum((x-u) ** 2)
    posteriorSyntax = T.sum(n * T.log(T.nnet.softmax(x))) + normalPdfSyntax
    posterior = theano.function(inputs=[x,u,sigma], outputs=posteriorSyntax)

    gPosteriorSyntax = T.grad(cost=posteriorSyntax, wrt=[x,sigma])
    gPosterior = theano.function(inputs=[x,u,sigma], outputs=gPosteriorSyntax)
    t0 = time.time()
    estimated_parameter = hamiltonianMonteCarlo(numpy.random.normal(0., 1., sample.shape[0]+1),callPosterior,callGPosterior)
    t1 = time.time()
    print 'Looping times took', t1 - t0, 'seconds'
    print softmax(estimated_parameter)



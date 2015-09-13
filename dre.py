from scipy.stats import binom
import pandas as pd
import numpy

def newton(num_theta,sampleEntry,sampleHospital):
    theta = numpy.zeros(num_theta)
    for _ in range(100):
        theta = df(theta,sampleEntry,sampleHospital)
        
def kernel(x,x2):
    numpy.exp(-(x-x2)**2/2*h**2)
    
def df(thetas,sampleEntry,sampleHospital):
    sum_basis_result = numpy.zeros(len(thetas))
    for x in sampleEntry:
        basis_result = []
        for idx in range(len(thetas)):
            basis_result.append(kernel(x,sample[idx]))
        sum_basis_result += basis_result
    sum_basis_result /= len(thetas)
    
    
    sum_basis_result2 = numpy.zeros(len(thetas))
    for x in sampleHospital:
        basis_result = []
        for idx in range(len(thetas)):
            basis_result.append(kernel(x,sample[idx]))
        sum_basis_result2 += basis_result / (thetas * basis_result)
    sum_basis_result2 /= len(thetas)
    
    
    return sum_basis_result - sum_basis_result2

def main():
    x0 = 0.0    #最初のxの値
    num = [x0]
    newton(x0, num)
    print num
    
sample_entry = [1] * 10 + [0] * 1000
sample_hospital = [1] * 10 + [0] * 1000

sample = pd.DataFrame({"entry":sample_entry,"hospital":sample_hospital})

print sample

import numpy as np

cdef extern from "cuda_kliep.h":
    ctypedef struct DTF:
        int n_kernel_fold
        
    ctypedef struct cudaKliep:
        pass
        
    DTF *diffTargetFunctionInit(float *basis_results, int n_basis_result, int n_kernel_fold)
    void diffTargetFunctionCalc(DTF *dtf, float *thetas, float *result)
    void diffTargetFunctionFree(DTF *dtf)
    void cudaKliepOpt(float *sum_basis_result, float *basis_result, int n_basis_result, int n_kernel_fold, int n_iter, float learning_rate, float* theta)
    cudaKliep *cudaKliepInit(int n_kernel_fold, float bandwidth, int n_iter, float learning_rate, float *init_theta)
    void cudaKliepFit(cudaKliep *cuda_kliep, float *sample_X, int n_sample_X, float *sample_Y, int n_sample_Y, int n_dimention, float *theta)
    void cudaKliepPredict(cudaKliep *cuda_kliep, float *sample_kernel, float *sample, int n_sample, int n_dimention, float *est_y)
    void cudaKliepFree(cudaKliep *cuda_kliep)
    
cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *)
    
cdef class cudaKliepWrap:
    cdef cudaKliep *cuda_kliep
    cdef int n_kernel_fold
    
    def __init__(self, n_kernel_fold, bandwidth, n_iter, learning_rate, init_theta):
        cdef float *init_theta_c = <float*>malloc(sizeof(float) * n_kernel_fold)
        for i in range(n_kernel_fold):
            init_theta_c[i] = init_theta[i]
            
        self.cuda_kliep = cudaKliepInit(n_kernel_fold, bandwidth, n_iter, learning_rate, init_theta_c)
        self.n_kernel_fold = n_kernel_fold
    
    def fit(self, sample_X, sample_Y):
        cdef int n_sample_X = sample_X.shape[0]
        cdef int n_sample_Y = sample_Y.shape[0]
        cdef int n_dimention = sample_X.shape[1]
        
        cdef float *c_sample_X = <float*>malloc(sizeof(float) * n_sample_X * n_dimention)
        cdef float *c_sample_Y = <float*>malloc(sizeof(float) * n_sample_Y * n_dimention)
        
        for i in range(n_sample_X):
            for j in range(n_dimention):
                c_sample_X[i * n_dimention + j] = sample_X[i, j]
                
        for i in range(n_sample_Y):
            for j in range(n_dimention):
                c_sample_Y[i * n_dimention + j] = sample_Y[i, j]
                
        cdef float *theta = <float*>malloc(sizeof(float) * self.n_kernel_fold)
        cudaKliepFit(self.cuda_kliep, c_sample_X, n_sample_X, c_sample_Y, n_sample_Y, n_dimention, theta);
        
        r_theta = []
        for i in range(self.n_kernel_fold):
            r_theta.append(theta[i])
        
        free(c_sample_X)
        free(c_sample_Y)
        free(theta)
        
        return np.array(r_theta)
    
    def predict(self, sample_X, sample_kernel):
        cdef int n_sample_X = sample_X.shape[0]
        cdef int n_dimention = sample_X.shape[1]
        
        cdef float *c_sample_X = <float*>malloc(sizeof(float) * n_sample_X * n_dimention)
        for i in range(n_sample_X):
            for j in range(n_dimention):
                c_sample_X[i * n_dimention + j] = sample_X[i, j]
        
        cdef int n_kernel_fold = sample_kernel.shape[0]
        cdef float *c_sample_kernel = <float*>malloc(sizeof(float) * n_kernel_fold * n_dimention)
        
        for i in range(n_kernel_fold):
            for j in range(n_dimention):
                c_sample_kernel[i * n_dimention + j] = sample_kernel[i, j]
                
        cdef float *c_est_y = <float*>malloc(sizeof(float) * n_sample_X)
        cudaKliepPredict(self.cuda_kliep, c_sample_kernel, c_sample_X, n_sample_X, n_dimention, c_est_y)
        
        est_y = []
        for i in range(n_sample_X):
            est_y.append(c_est_y[i])
        
        free(c_est_y)
        free(c_sample_X)
        free(c_sample_kernel)
        
        return np.array(est_y)
    
    def __dealloc__(self):
        cudaKliepFree(self.cuda_kliep)
        
"""
def cudaKliepOptWrap(sum_basis_result, basis_results, n_iter, learning_rate, theta):
    cdef int n_basis_results = basis_results.shape[0]
    cdef int n_kernel_fold = basis_results.shape[1]
    cdef float *c_basis_results = <float*>malloc(sizeof(float) * n_basis_results * n_kernel_fold)
    cdef float *c_theta = <float*>malloc(sizeof(float) * n_kernel_fold)
    
    for i in range(n_basis_results):
        for j in range(n_kernel_fold):
            c_basis_results[i * n_kernel_fold + j] = <float>basis_results[i, j]
    
    cdef float *c_sum_basis_result = <float*>malloc(sizeof(float) * n_kernel_fold)
    
    for i in range(n_kernel_fold):
        c_sum_basis_result[i] = <float>sum_basis_result[i]
    
    for i in range(n_kernel_fold):
        c_theta[i] = theta[i]
        
    cudaKliepOpt(c_sum_basis_result, c_basis_results, n_basis_results, n_kernel_fold, n_iter, learning_rate, c_theta)
    
    for i in range(n_kernel_fold):
        theta[i] = c_theta[i]
    
    free(c_theta)
    free(c_basis_results)
    free(c_sum_basis_result)
    
    return theta
"""

"""
cdef class DiffTargetFunction:
    cdef DTF *dtf
    
    def __dealloc__(self):
        diffTargetFunctionFree(self.dtf)
        
    def __init__(self, basis_results):
        cdef int n_basis_results = basis_results.shape[0]
        cdef int n_kernel_fold = basis_results.shape[1]
        
        print "n_basis_results", n_basis_results
        print "n_kernel_fold", n_kernel_fold
        
        cdef float *c_basis_results = <float*>malloc(sizeof(float) * n_basis_results * n_kernel_fold)
        
        for i in range(n_basis_results):
            for j in range(n_kernel_fold):
                c_basis_results[i * n_kernel_fold + j] = <float>basis_results[i, j]
                
        #self.dtf = diffTargetFunctionInit(c_basis_results, n_basis_results, n_kernel_fold)
    
    def calc(self, theta):
        cdef float *c_theta = <float*>malloc(sizeof(float) * self.dtf.n_kernel_fold)
        cdef float *result = <float*>malloc(sizeof(float) * self.dtf.n_kernel_fold)
        for i in range(self.dtf.n_kernel_fold):
            c_theta[i] = theta[i]
            
        diffTargetFunctionCalc(self.dtf, c_theta, result)
        
        delta = []
        for i in range(self.dtf.n_kernel_fold):
            delta.append(result[i])
            
        free(c_theta)
        
        return np.array(delta)
"""

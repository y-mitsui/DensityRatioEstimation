#include <cuda_runtime.h>
#include <cublas.h>
#include <assert.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <time.h>
#include "cuda_kliep.h"

#define d_cudaMalloc(ptr, size) debug_cudaMalloc(ptr, size, __FILE__, __LINE__)

static void debug_cudaMalloc(void **ptr, size_t size, const char *file, int line_no){
    cudaError_t status;
    status = cudaMalloc(ptr, size);
    if (status != cudaSuccess){
        fprintf(stderr, "cudaMalloc error %s:%d\n", file, line_no);
        exit(1);
    }
}

__global__ void v_add_div_scaler(float *v_R, float *v_A, float *dev_scaler, int max_iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        v_R[tid] += v_A[tid] / dev_scaler[0];
        
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void v_div_scaler(float *v_A, float scaler, int max_iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        v_A[tid] /= scaler;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void  v_mul_scaler_sub(float *v_A, float *v_B, float scaler, int max_iter) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        v_A[tid] -= v_B[tid] * scaler;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void  v_mul_scaler_sub_r(float *v_R, float *v_A, float *v_B, float scaler, int max_iter) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        
        v_R[tid] = v_A[tid] - v_B[tid] * scaler;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void  v_sub(float *v_A, float *v_B, int max_iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        v_A[tid] -= v_B[tid];
        tid += blockDim.x * gridDim.x;
    }
}

DTF *diffTargetFunctionInit(float *dev_sum_basis_result, float *dev_basis_results, int n_basis_result, int n_kernel_fold) {
    DTF *dtf = (DTF*)malloc(sizeof(DTF));
    
    dtf->dev_basis_results = dev_basis_results;
    dtf->dev_sum_basis_result = dev_sum_basis_result;
    
    dtf->n_basis_result = n_basis_result;
    dtf->n_kernel_fold = n_kernel_fold;
    cublasStatus_t blas_status = cublasCreate(&dtf->blas_handle);
    assert( blas_status == CUBLAS_STATUS_SUCCESS);
    return dtf;
}

void diffTargetFunctionFree(DTF *dtf) {
    cublasDestroy(dtf->blas_handle);
    free(dtf);
}

void diffTargetFunctionCalc(DTF *dtf, float *dev_theta, float *result, float learning_rate) {
    int n_thread = 512;
    float alpha = 1.0f;
    float beta  = 0.0f;
    float *dev_temp, *sum_basis_result2;
    
    d_cudaMalloc((void**)&dev_temp, sizeof(float));
    d_cudaMalloc((void**)&sum_basis_result2, sizeof(float) * dtf->n_kernel_fold);
    cudaMemset(sum_basis_result2, 0, sizeof(float) * dtf->n_kernel_fold);
    
    int n_iter = dtf->n_kernel_fold;
    int n_block = (n_iter + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;
    for(int i=0; i < dtf->n_basis_result; i++) {
        cublasStatus_t blas_status = cublasSgemm(dtf->blas_handle, CUBLAS_OP_N,  CUBLAS_OP_N, 1, 1, dtf->n_kernel_fold, 
                                    &alpha, dev_theta, 1,
                                    &dtf->dev_basis_results[i * dtf->n_kernel_fold], dtf->n_kernel_fold, 
                                    &beta, dev_temp, 1);
        assert( blas_status == CUBLAS_STATUS_SUCCESS);
        
        v_add_div_scaler<<<n_block, n_thread>>>(sum_basis_result2, &dtf->dev_basis_results[i * dtf->n_kernel_fold], dev_temp, n_iter);
    }
    v_mul_scaler_sub_r<<<n_block, n_thread>>>(result, dtf->dev_sum_basis_result, sum_basis_result2, 1.0 / (float)dtf->n_basis_result, n_iter);
    //cudaMemcpy(result, sum_basis_result2, sizeof(float) * dtf->n_kernel_fold, cudaMemcpyDeviceToHost);
    //v_mul_scaler_sub_r<<<n_block, n_thread>>>(dev_theta, dev_theta, result, (float)learning_rate, n_iter);
    
    //cudaFree(dev_theta);
    cudaFree(dev_temp);
    cudaFree(sum_basis_result2);
}

__global__ void pow2Sum(float *sample_X, float *sample_kernel_fold, int n_dimention, float mul_scaler, float mul_scaler2, float *result, int max_iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        float temp = 0.0;
        for(int i=0; i < n_dimention; i++) {
            //printf("sample_X[%d]:%f sample_kernel_fold[%d]:%f\n", tid, sample_X[i], tid * n_dimention + i, sample_kernel_fold[tid * n_dimention + i]);
            temp += (sample_X[i] - sample_kernel_fold[tid * n_dimention + i]) * (sample_X[i] - sample_kernel_fold[tid * n_dimention + i]);
        }
        result[tid] += expf(-temp * mul_scaler2) * mul_scaler;
        //printf("result[%d]:%f %f %f\n", tid, result[tid], temp, mul_scaler);
        tid += blockDim.x * gridDim.x;
    }
}

void cudaKliepKernel(cudaKliep *cuda_kliep, float *sample_kernel_fold, float *sample, int n_dimention, float *dev_result, float mul_scaler) {
    float mul_scaler2 = 1.0 / (2 * cuda_kliep->bandwidth * cuda_kliep->bandwidth);
    int n_thread = 512;
    int n_iter = cuda_kliep->n_kernel_fold;
    int n_block = (n_iter + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;
    
    pow2Sum<<<n_block, n_thread>>>(sample, sample_kernel_fold, n_dimention, mul_scaler, mul_scaler2, dev_result, n_iter);
}

cudaKliep *cudaKliepInit(int n_kernel_fold, float bandwidth, int n_iter, float learning_rate, float *init_theta) {
    cudaKliep *cuda_kliep = (cudaKliep*)malloc(sizeof(cudaKliep));
    cuda_kliep->n_kernel_fold = n_kernel_fold;
    d_cudaMalloc((void**)&cuda_kliep->dev_theta, sizeof(float) * cuda_kliep->n_kernel_fold);
    cudaMemcpy(cuda_kliep->dev_theta, init_theta, sizeof(float) * cuda_kliep->n_kernel_fold, cudaMemcpyHostToDevice);
    cuda_kliep->bandwidth = bandwidth;
    cuda_kliep->n_iter = n_iter;
    cuda_kliep->learning_rate = learning_rate;
    return cuda_kliep;
}

void cudaKliepFree(cudaKliep *cuda_kliep) {
    cudaFree(cuda_kliep->dev_theta);
    free(cuda_kliep);
}

void cudaKliepFit(cudaKliep *cuda_kliep, float *sample_X, int n_sample_X, float *sample_Y, int n_sample_Y, int n_dimention, float *theta) {
    float *dev_sample_X, *dev_sample_Y;
    
    if (cuda_kliep->n_kernel_fold < 0) cuda_kliep->n_kernel_fold = n_sample_X;
    d_cudaMalloc((void**)&dev_sample_X, sizeof(float) * n_sample_X * n_dimention);
    cudaMemcpy(dev_sample_X, sample_X, sizeof(float) * n_sample_X * n_dimention, cudaMemcpyHostToDevice);
    
    d_cudaMalloc((void**)&dev_sample_Y, sizeof(float) * n_sample_Y * n_dimention);
    cudaMemcpy(dev_sample_Y, sample_Y, sizeof(float) * n_sample_Y * n_dimention, cudaMemcpyHostToDevice);
    
    float *dev_sum_basis_result;
    d_cudaMalloc((void**)&dev_sum_basis_result, sizeof(float) * cuda_kliep->n_kernel_fold);
    cudaMemset(dev_sum_basis_result, 0, sizeof(float) * cuda_kliep->n_kernel_fold);
    for (int i=0; i < n_sample_Y; i++) {
        cudaKliepKernel(cuda_kliep, dev_sample_X, &dev_sample_Y[i * n_dimention], n_dimention, dev_sum_basis_result, 1.0 / (float)n_sample_Y);
    }
    cudaFree(dev_sample_Y);
    
    float *dev_basis_result;
    d_cudaMalloc((void**)&dev_basis_result, sizeof(float) * cuda_kliep->n_kernel_fold * n_sample_X);
    cudaMemset(dev_basis_result, 0, sizeof(float) * cuda_kliep->n_kernel_fold * n_sample_X);
    for (int i=0; i < n_sample_X; i++) {
        cudaKliepKernel(cuda_kliep, dev_sample_X, &dev_sample_X[i * n_dimention], n_dimention, &dev_basis_result[i * cuda_kliep->n_kernel_fold], 1.0);
    }
    cudaFree(dev_sample_X);
    
    cudaKliepOpt(cuda_kliep, dev_sum_basis_result, dev_basis_result, n_sample_X);
    
    cudaFree(dev_sum_basis_result);
    cudaFree(dev_basis_result);
    
    cudaMemcpy(theta, cuda_kliep->dev_theta, sizeof(float) * cuda_kliep->n_kernel_fold, cudaMemcpyDeviceToHost);
}

__global__ void thresoldNagative(float *sample_X, int max_iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < max_iter){
        if (sample_X[tid] < 0) sample_X[tid] = 0;
        tid += blockDim.x * gridDim.x;
    }
}

void cudaKliepOpt(cudaKliep *cuda_kliep, float *dev_sum_basis_result, float *dev_basis_result, int n_basis_result) {
    float *delta_theta;
    float *theta = (float *)malloc(sizeof(float) * cuda_kliep->n_kernel_fold);
    DTF *dtf = diffTargetFunctionInit(dev_sum_basis_result, dev_basis_result, n_basis_result, cuda_kliep->n_kernel_fold);
    
    int n_thread = 512;
    int max_iter = dtf->n_kernel_fold;
    int n_block = (max_iter + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;
    
    d_cudaMalloc((void**)&delta_theta, sizeof(float) * dtf->n_kernel_fold);
    time_t t1 = time(NULL);
    
    for (int i=0; i < cuda_kliep->n_iter; i++) {
        diffTargetFunctionCalc(dtf, cuda_kliep->dev_theta, delta_theta, cuda_kliep->learning_rate);
        v_mul_scaler_sub<<<n_block, n_thread>>>(cuda_kliep->dev_theta, delta_theta, (float)cuda_kliep->learning_rate, max_iter);
        thresoldNagative<<<n_block, n_thread>>>(cuda_kliep->dev_theta, dtf->n_kernel_fold);
        if (i % 500 == 0){
            printf("%d / %d (%dsec)\n", i , cuda_kliep->n_iter, time(NULL) - t1);
            t1 = time(NULL);
            
            cudaMemcpy(theta, cuda_kliep->dev_theta, sizeof(float) * dtf->n_kernel_fold, cudaMemcpyDeviceToHost);
            int n_view = (dtf->n_kernel_fold < 15) ? dtf->n_kernel_fold : 15;
            for(int j=0;j < n_view; j++) {
                printf("%f ", j , theta[j]);
            }
            puts("");
        }
        
    }
    diffTargetFunctionFree(dtf);
    cudaFree(delta_theta);
    free(theta);
}

void cudaKliepPredict(cudaKliep *cuda_kliep, float *sample_kernel, float *sample, int n_sample, int n_dimention, float *est_y) {
    float *dev_temp, *dev_kernel, *dev_sample, *dev_sample_kernel;
    
    cublasHandle_t blas_handle;
    cublasStatus_t blas_status = cublasCreate(&blas_handle);
    assert( blas_status == CUBLAS_STATUS_SUCCESS);
    
    d_cudaMalloc((void**)&dev_temp, sizeof(float));
    d_cudaMalloc((void**)&dev_kernel, sizeof(float) * cuda_kliep->n_kernel_fold);
    d_cudaMalloc((void**)&dev_sample, sizeof(float) * n_sample * n_dimention);
    d_cudaMalloc((void**)&dev_sample_kernel, sizeof(float) * cuda_kliep->n_kernel_fold * n_dimention);
    cudaMemcpy(dev_sample, sample, sizeof(float) * n_sample * n_dimention, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sample_kernel, sample_kernel, sizeof(float) * cuda_kliep->n_kernel_fold * n_dimention, cudaMemcpyHostToDevice);
    
    float alpha = 1.0f;
    float beta  = 0.0f;
    for (int i=0; i < n_sample; i++) {
        cudaMemset(dev_kernel, 0, sizeof(float) * cuda_kliep->n_kernel_fold);
        cudaKliepKernel(cuda_kliep, dev_sample_kernel, &dev_sample[i * n_dimention], n_dimention, dev_kernel, 1.0);
        cublasStatus_t blas_status = cublasSgemm(blas_handle, CUBLAS_OP_N,  CUBLAS_OP_N, 1, 1, cuda_kliep->n_kernel_fold, 
                                    &alpha, cuda_kliep->dev_theta, 1,
                                    dev_kernel, cuda_kliep->n_kernel_fold, 
                                    &beta, dev_temp, 1);
        assert( blas_status == CUBLAS_STATUS_SUCCESS);
        
        cudaMemcpy(&est_y[i], dev_temp, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(dev_temp);
    cudaFree(dev_kernel);
    cudaFree(dev_sample);
    cudaFree(dev_sample_kernel);
    cublasDestroy(blas_handle);
}


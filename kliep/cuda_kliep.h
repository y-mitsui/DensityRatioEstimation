#include <cublas.h>
typedef struct {
    float *dev_basis_results;
    float *dev_sum_basis_result;
    int n_basis_result;
    int n_kernel_fold;
    cublasHandle_t blas_handle;
}DTF;

typedef struct {
    int n_kernel_fold;
    int n_iter;
    float bandwidth;
    float learning_rate;
    float *dev_theta;
}cudaKliep;

DTF *diffTargetFunctionInit(float *basis_results, int n_basis_result, int n_kernel_fold);
void diffTargetFunctionCalc(DTF *dtf, float *thetas, float *result);
void diffTargetFunctionFree(DTF *dtf);
void cudaKliepOpt(cudaKliep *cuda_kliep, float *dev_sum_basis_result, float *dev_basis_result, int n_basis_result);
cudaKliep *cudaKliepInit(int n_kernel_fold, float bandwidth, int n_iter, float learning_rate, float *init_theta);
void cudaKliepFit(cudaKliep *cuda_kliep, float *sample_X, int n_sample_X, float *sample_Y, int n_sample_Y, int n_dimention, float *theta);
void cudaKliepPredict(cudaKliep *cuda_kliep, float *sample_kernel, float *sample, int n_sample, int n_dimention, float *est_y);
void cudaKliepFree(cudaKliep *cuda_kliep);


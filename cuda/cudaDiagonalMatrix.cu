#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


__global__ void matrixDiagonal(float *A, float *sum, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < column) {
        *sum += A[row * N + column];
    }
 }


int main(int argc, char *argv[]) {
    int N = 8;
    size_t matrixSize = sizeof(float) * N * N;
    size_t floatSize = sizeof(float);

    // Allocate matrix on CPU
    float *h_matrix_A = (float *)malloc(matrixSize);
    for (size_t i = 0; i < N * N; i ++) {
        h_matrix_A[i] = rand() / (float)RAND_MAX;
    }

    float *h_sum = (float *)calloc(1, floatSize);

    cudaError_t err = cudaSuccess;
    // Allocate memory to GPU Device
    float *d_matrix_A = NULL;
    err = cudaMalloc((void **)&d_matrix_A, matrixSize);

    float *d_sum = NULL;
    err = cudaMalloc((void **)&d_sum, floatSize);

    // Copy data from CPU to GPU
    err = cudaMemcpy(d_matrix_A, h_matrix_A, matrixSize, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_sum, h_sum, floatSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    matrixDiagonal<<<1, threadsPerBlock>>>(d_matrix_A, d_sum, N);
    err = cudaGetLastError();

    // Copy result from GPU to CPU
    err = cudaMemcpy(h_sum, d_sum, floatSize, cudaMemcpyDeviceToHost);

    printf("Result of sum: %d\n", *h_sum);

    // Free memory
    err = cudaFree(d_matrix_A);
    err = cudaFree(d_sum);

    free(h_matrix_A);
    free(h_sum);

    return 0;
}

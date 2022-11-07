// C++ Libs
#include <iostream>
// CUDA libs
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;


#define CUDA_CHECK(call)                                                        \
    if ((call) != cudaSuccess) {                                                \
        cudaError_t err = cudaGetLastError();                                   \
        cerr << "CUDA error calling \"" #call "\", code is " << err << "\n";    \
        exit(1);                                                                \
    }


void printMatrix(float *matrix, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            cout << matrix[i * N + j] << ' ';
        }
        cout << '\n';
    }
}


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
        // h_matrix_A[i] = rand() / (float)RAND_MAX;
        h_matrix_A[i] = 1;  // debug
    }
    printMatrix(h_matrix_A, N);

    float *h_sum = (float *)calloc(1, floatSize);

    // Allocate memory to GPU Device
    float *d_matrix_A = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_matrix_A, matrixSize));

    float *d_sum = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_sum, floatSize));

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_matrix_A, h_matrix_A, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum, h_sum, floatSize, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(N / 2, N / 2);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    matrixDiagonal<<<numBlocks, threadsPerBlock>>>(d_matrix_A, d_sum, N);

    // Copy result from GPU to CPU
    CUDA_CHECK(cudaMemcpy(h_sum, d_sum, floatSize, cudaMemcpyDeviceToHost));

    cout << "Result of sum:" << *h_sum << "\n";

    // Free memory
    CUDA_CHECK(cudaFree(d_matrix_A));
    CUDA_CHECK(cudaFree(d_sum));

    free(h_matrix_A);
    free(h_sum);

    return 0;
}

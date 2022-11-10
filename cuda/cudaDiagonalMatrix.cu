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

    for (int i = 0; i < N; i++) {
        if (row < i) {
            sum[row] += A[row * N + i];
        }  
    }
 }


int main(int argc, char *argv[]) {
    int N = 8;
    size_t matrixSize = sizeof(float) * N * N;
    size_t vectorSize = sizeof(float) * N;

    // Allocate matrix on CPU
    float *h_matrix_A = (float *)malloc(matrixSize);
    for (size_t i = 0; i < N * N; i++) {
        // h_matrix_A[i] = rand() / (float)RAND_MAX;
        h_matrix_A[i] = 1;  // debug
    }
    printMatrix(h_matrix_A, N);

    float *h_vector_sum = (float *)calloc(N, sizeof(float));

    // Allocate memory to GPU Device
    float *d_matrix_A = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_matrix_A, matrixSize));

    float *d_vector_sum = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_vector_sum, vectorSize));

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_matrix_A, h_matrix_A, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vector_sum, h_vector_sum, vectorSize, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(N / 2);
    dim3 numBlocks(N / threadsPerBlock.x);
    matrixDiagonal<<<numBlocks, threadsPerBlock>>>(d_matrix_A, d_vector_sum, N);

    // Copy result from GPU to CPU
    CUDA_CHECK(cudaMemcpy(h_vector_sum, d_vector_sum, vectorSize, cudaMemcpyDeviceToHost));

    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_vector_sum[i];
    }

    cout << "Result of sum:" << sum << "\n";

    // Free memory
    CUDA_CHECK(cudaFree(d_matrix_A));
    CUDA_CHECK(cudaFree(d_vector_sum));

    free(h_matrix_A);
    free(h_vector_sum);

    return 0;
}

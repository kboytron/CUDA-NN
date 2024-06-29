#include <stdio.h>

__global__ void matmul(const float* A, const float* B, float* C, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < m) {
        float value = 0;
        for (int i = 0; i < k; ++i) {
            value += A[row * k + i] * B[i * m + col];
        }
        C[row * m + col] = value;
    }
}

__global__ void relu(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

__global__ void softmax(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += expf(Z[i]);
        }
        A[idx] = expf(Z[idx]) / sum_exp;
    }
}

#include <cuda_runtime.h>

__global__ void reluKernel(float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Z[idx] = fmaxf(Z[idx], 0.0f);
    }
}

__global__ void softmaxKernel(float* Z, float* A, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float max_val = -INFINITY;
        for (int i = 0; i < rows; ++i) {
            max_val = fmaxf(max_val, Z[i * cols + col]);
        }
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            float exp_val = expf(Z[i * cols + col] - max_val);
            A[i * cols + col] = exp_val;
            sum += exp_val;
        }
        for (int i = 0; i < rows; ++i) {
            A[i * cols + col] /= sum;
        }
    }
}

__global__ void forwardPropKernel(float* W, float* b, float* X, float* Z, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += W[row * n + i] * X[i * k + col];
        }
        Z[row * k + col] = sum + b[row];
    }
}

__global__ void reluDerivKernel(float* Z, float* dZ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (Z[idx] > 0) ? dZ[idx] : 0;
    }
}

__global__ void backwardPropKernel(float* dZ, float* A, float* dW, float* db, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += dZ[row * k + i] * A[col * k + i];
        }
        dW[row * n + col] = sum / k;
    }
    if (col == 0 && row < m) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += dZ[row * k + i];
        }
        db[row] = sum / k;
    }
}

extern "C" {
    __declspec(dllexport) void launchReluKernel(float* d_Z, int size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        reluKernel << <numBlocks, blockSize >> > (d_Z, size);
    }

    __declspec(dllexport) void launchSoftmaxKernel(float* d_Z, float* d_A, int rows, int cols) {
        int blockSize = 256;
        int numBlocks = (cols + blockSize - 1) / blockSize;
        softmaxKernel << <numBlocks, blockSize >> > (d_Z, d_A, rows, cols);
    }

    __declspec(dllexport) void launchForwardPropKernel(float* d_W, float* d_b, float* d_X, float* d_Z, int m, int n, int k) {
        dim3 blockSize(16, 16);
        dim3 numBlocks((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        forwardPropKernel << <numBlocks, blockSize >> > (d_W, d_b, d_X, d_Z, m, n, k);
    }

    __declspec(dllexport) void launchReluDerivKernel(float* d_Z, float* d_dZ, int size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        reluDerivKernel << <numBlocks, blockSize >> > (d_Z, d_dZ, size);
    }

    __declspec(dllexport) void launchBackwardPropKernel(float* d_dZ, float* d_A, float* d_dW, float* d_db, int m, int n, int k) {
        dim3 blockSize(16, 16);
        dim3 numBlocks((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        backwardPropKernel << <numBlocks, blockSize >> > (d_dZ, d_A, d_dW, d_db, m, n, k);
    }
}
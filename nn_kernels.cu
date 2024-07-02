#include <cuda_runtime.h>

__global__ void reluKernel(float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Z[idx] = fmaxf(Z[idx], 0.0f);
    }
}

extern "C" {
    __declspec(dllexport) void launchReluKernel(float* d_Z, int size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        reluKernel<<<numBlocks, blockSize>>>(d_Z, size);
        cudaDeviceSynchronize();
    }
}
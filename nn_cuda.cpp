#include <cuda_runtime.h>
#include "nn_kernels.cu"

extern "C" {
    __declspec(dllexport) void relu(float* h_Z, int size) {
        float* d_Z;
        cudaMalloc(&d_Z, size * sizeof(float));
        cudaMemcpy(d_Z, h_Z, size * sizeof(float), cudaMemcpyHostToDevice);

        launchReluKernel(d_Z, size);

        cudaMemcpy(h_Z, d_Z, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_Z);
    }
}
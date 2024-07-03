#include <cuda_runtime.h>

extern "C" {
    void launchReluKernel(float* d_Z, int size);
    void launchSoftmaxKernel(float* d_Z, float* d_A, int rows, int cols);
    void launchForwardPropKernel(float* d_W, float* d_b, float* d_X, float* d_Z, int m, int n, int k);
    void launchReluDerivKernel(float* d_Z, float* d_dZ, int size);
    void launchBackwardPropKernel(float* d_dZ, float* d_A, float* d_dW, float* d_db, int m, int n, int k);

    __declspec(dllexport) void relu(float* h_Z, int size) {
        float* d_Z;
        cudaMalloc(&d_Z, size * sizeof(float));
        cudaMemcpy(d_Z, h_Z, size * sizeof(float), cudaMemcpyHostToDevice);

        launchReluKernel(d_Z, size);

        cudaMemcpy(h_Z, d_Z, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_Z);
    }

    __declspec(dllexport) void softmax(float* h_Z, float* h_A, int rows, int cols) {
        float* d_Z, * d_A;
        cudaMalloc(&d_Z, rows * cols * sizeof(float));
        cudaMalloc(&d_A, rows * cols * sizeof(float));
        cudaMemcpy(d_Z, h_Z, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

        launchSoftmaxKernel(d_Z, d_A, rows, cols);

        cudaMemcpy(h_A, d_A, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_Z);
        cudaFree(d_A);
    }

    __declspec(dllexport) void forwardProp(float* h_W, float* h_b, float* h_X, float* h_Z, int m, int n, int k) {
        float* d_W, * d_b, * d_X, * d_Z;
        cudaMalloc(&d_W, m * n * sizeof(float));
        cudaMalloc(&d_b, m * sizeof(float));
        cudaMalloc(&d_X, n * k * sizeof(float));
        cudaMalloc(&d_Z, m * k * sizeof(float));
        cudaMemcpy(d_W, h_W, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, h_X, n * k * sizeof(float), cudaMemcpyHostToDevice);

        launchForwardPropKernel(d_W, d_b, d_X, d_Z, m, n, k);

        cudaMemcpy(h_Z, d_Z, m * k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_X);
        cudaFree(d_Z);
    }

    __declspec(dllexport) void reluDeriv(float* h_Z, float* h_dZ, int size) {
        float* d_Z, * d_dZ;
        cudaMalloc(&d_Z, size * sizeof(float));
        cudaMalloc(&d_dZ, size * sizeof(float));
        cudaMemcpy(d_Z, h_Z, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dZ, h_dZ, size * sizeof(float), cudaMemcpyHostToDevice);

        launchReluDerivKernel(d_Z, d_dZ, size);

        cudaMemcpy(h_dZ, d_dZ, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_Z);
        cudaFree(d_dZ);
    }

    __declspec(dllexport) void backwardProp(float* h_dZ, float* h_A, float* h_dW, float* h_db, int m, int n, int k) {
        float* d_dZ, * d_A, * d_dW, * d_db;
        cudaMalloc(&d_dZ, m * k * sizeof(float));
        cudaMalloc(&d_A, n * k * sizeof(float));
        cudaMalloc(&d_dW, m * n * sizeof(float));
        cudaMalloc(&d_db, m * sizeof(float));
        cudaMemcpy(d_dZ, h_dZ, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, h_A, n * k * sizeof(float), cudaMemcpyHostToDevice);

        launchBackwardPropKernel(d_dZ, d_A, d_dW, d_db, m, n, k);

        cudaMemcpy(h_dW, d_dW, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_db, d_db, m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_dZ);
        cudaFree(d_A);
        cudaFree(d_dW);
        cudaFree(d_db);
    }
}
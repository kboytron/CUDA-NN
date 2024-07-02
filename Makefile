NVCC = nvcc
CFLAGS = -O3 -arch=sm_50
LDFLAGS = -shared

all: nn_cuda.dll

nn_cuda.dll: nn_cuda.cpp nn_kernels.cu
    $(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
    del nn_cuda.dll
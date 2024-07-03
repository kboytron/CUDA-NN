NVCC = nvcc
CFLAGS = -O3 -arch=sm_50
LDFLAGS = -shared
PYTHON = python

all: nn_cuda.dll

nn_cuda.dll: nn_cuda.cpp nn_kernels.cu
    $(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ nn_cuda.cpp nn_kernels.cu

run: nn_cuda.dll
    $(PYTHON) nn.py

clean:
    del nn_cuda.dll

.PHONY: all run clean
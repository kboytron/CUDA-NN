import numpy as np
from sklearn.datasets import fetch_openml
import ctypes

# Load the compiled CUDA kernel
cuda_lib = ctypes.CDLL('./cuda_kernels.so')

# Define the argument types for the CUDA functions
cuda_lib.matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

cuda_lib.relu.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

cuda_lib.softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

def to_device(np_array):
    c_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return c_array

def matmul(A, B, C, n, m, k):
    cuda_lib.matmul(A, B, C, n, m, k)

def relu(Z, A, size):
    cuda_lib.relu(Z, A, size)

def softmax(Z, A, size):
    cuda_lib.softmax(Z, A, size)

def init_params():
    W1 = np.random.rand(10, 784).astype(np.float32) - 0.5
    b1 = np.random.rand(10, 1).astype(np.float32) - 0.5
    W2 = np.random.rand(10, 10).astype(np.float32) - 0.5
    b2 = np.random.rand(10, 1).astype(np.float32) - 0.5
    return W1, b1, W2, b2

def load_and_preprocess_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X = X / 255.0
    y = y.astype(int)
    Y = np.eye(10)[y]
    X_train, X_test = X[:60000], X[60000:]
    Y_train, Y_test = Y[:60000], Y[60000:]
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_and_preprocess_data()
W1, b1, W2, b2 = init_params()

def forward_prop(W1, b1, W2, b2, X):
    n, m = W1.shape
    k = X.shape[0]

    Z1 = np.zeros((n, X.shape[1]), dtype=np.float32)
    A1 = np.zeros_like(Z1)
    Z2 = np.zeros((W2.shape[0], X.shape[1]), dtype=np.float32)
    A2 = np.zeros_like(Z2)

    W1_ctypes = to_device(W1)
    b1_ctypes = to_device(b1)
    W2_ctypes = to_device(W2)
    b2_ctypes = to_device(b2)
    X_ctypes = to_device(X)
    Z1_ctypes = to_device(Z1)
    A1_ctypes = to_device(A1)
    Z2_ctypes = to_device(Z2)
    A2_ctypes = to_device(A2)

    matmul(W1_ctypes, X_ctypes, Z1_ctypes, n, X.shape[1], m)
    Z1 += b1
    relu(Z1_ctypes, A1_ctypes, Z1.size)
    matmul(W2_ctypes, A1_ctypes, Z2_ctypes, W2.shape[0], A1.shape[1], W2.shape[1])
    Z2 += b2
    softmax(Z2_ctypes, A2_ctypes, Z2.size)

    return Z1, A1, Z2, A2

Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train.T)
print("Forward propagation result:", A2)

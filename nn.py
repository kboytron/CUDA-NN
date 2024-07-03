import numpy as np
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import ctypes

cudaLib = ctypes.CDLL('./nn_cuda.dll')

cudaLib.relu.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int]
cudaLib.relu.restype = None

cudaLib.softmax.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_int]
cudaLib.softmax.restype = None

cudaLib.forwardProp.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_int, ctypes.c_int]
cudaLib.forwardProp.restype = None

cudaLib.reluDeriv.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int]
cudaLib.reluDeriv.restype = None

cudaLib.backwardProp.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_int, ctypes.c_int]
cudaLib.backwardProp.restype = None

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist['data']
target = mnist['target'].astype(np.int64)

# Convert to numpy array
data = np.array(data)
target = np.array(target)

# Combine data and target for shuffling
combined = np.c_[target, data]
np.random.shuffle(combined)

# Split the combined data
m, n = combined.shape
dataDev = combined[:1000].T
yDev = dataDev[0]
XDev = dataDev[1:n]
XDev = XDev / 255.0

dataTrain = combined[1000:m].T
yTrain = dataTrain[0]
xTrain = dataTrain[1:n]
xTrain = xTrain / 255.0

_, mTrain = xTrain.shape


def initParams():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2


def softmax(Z):
    A = np.zeros_like(Z, dtype=np.float32)
    cudaLib.softmax(Z.astype(np.float32), A, Z.shape[0], Z.shape[1])
    return A


def forwardProp(W1, b1, W2, b2, X):
    Z1 = np.zeros((W1.shape[0], X.shape[1]), dtype=np.float32)
    cudaLib.forwardProp(W1.astype(np.float32), b1.astype(np.float32), X.astype(np.float32), Z1, W1.shape[0], W1.shape[1], X.shape[1])
    A1 = relu(Z1)
    Z2 = np.zeros((W2.shape[0], A1.shape[1]), dtype=np.float32)
    cudaLib.forwardProp(W2.astype(np.float32), b2.astype(np.float32), A1.astype(np.float32), Z2, W2.shape[0], W2.shape[1], A1.shape[1])
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def relu(Z):
    Z_flat = Z.ravel().astype(np.float32)
    cudaLib.relu(Z_flat, Z_flat.size)
    return Z_flat.reshape(Z.shape)


def reluDeriv(Z):
    dZ = np.ones_like(Z, dtype=np.float32)
    cudaLib.reluDeriv(Z.astype(np.float32), dZ, Z.size)
    return dZ


def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY


def backwardProp(Z1, A1, Z2, A2, W1, W2, X, oneHotY, mTrain):
    dZ2 = A2 - oneHotY
    dW2 = 1 / mTrain * dZ2.dot(A1.T)
    db2 = 1 / mTrain * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * reluDeriv(Z1)
    dW1 = 1 / mTrain * dZ1.dot(X.T)
    db1 = 1 / mTrain * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2    
    return W1, b1, W2, b2


def getPredictions(A2):
    return np.argmax(A2, 0)


def getAccuracy(predictions, Y):
    return np.mean(predictions == Y)


def gradientDescent(X, Y, alpha, iterations, threshold=0.025):
    W1, b1, W2, b2 = initParams()
    oneHotY = oneHot(Y)
    prev_accuracy = 0
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardProp(Z1, A1, Z2, A2, W1, W2, X, oneHotY, mTrain)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            predictions = getPredictions(A2)
            accuracy = getAccuracy(predictions, Y)
            print(f"Iteration: {i}, Accuracy: {accuracy}")
            if abs(accuracy - prev_accuracy) < threshold:
                print("Early stopping as accuracy change is below threshold")
                break
            prev_accuracy = accuracy
    return W1, b1, W2, b2


def makePredictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions


def testPrediction(index, W1, b1, W2, b2):
    currentImage = xTrain[:, index, None]
    prediction = makePredictions(currentImage, W1, b1, W2, b2)
    label = yTrain[index]
    print(f"Prediction: {prediction}, Label: {label}")
    
    currentImage = currentImage.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(currentImage, interpolation='nearest')
    plt.show()

# Training the model
W1, b1, W2, b2 = gradientDescent(xTrain, yTrain, 0.10, 1500)

# Testing predictions
# testPrediction(0, W1, b1, W2, b2)
# testPrediction(1, W1, b1, W2, b2)
# testPrediction(2, W1, b1, W2, b2)
# testPrediction(3, W1, b1, W2, b2)

# Evaluating on the dev set
devPredictions = makePredictions(XDev, W1, b1, W2, b2)
print("Dev set accuracy:", getAccuracy(devPredictions, yDev))

import numpy as np

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return X > 0

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

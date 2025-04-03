import numpy as np

def flatten(X):
    if len(X.shape) == 4:
        return X.reshape(X.shape[0], -1)
    return X.reshape(1, -1)

import numpy as np
from .activations import relu_derivative

import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def backward_pass(self, X, Y, learning_rate):
    m = X.shape[0]

    if len(Y.shape) == 1:
        one_hot_y = np.zeros((m, self.output_size))
        one_hot_y[np.arange(m), Y] = 1
    else:
        one_hot_y = Y

    dZ3 = self.A3 - one_hot_y
    dW3 = np.dot(self.A2.T, dZ3) / m
    dB3 = np.sum(dZ3, axis=0) / m

    dZ2 = np.dot(dZ3, self.W3.T) * relu_derivative(self.A2)
    dW2 = np.dot(self.F1.T, dZ2) / m
    dB2 = np.sum(dZ2, axis=0) / m

    dF1 = np.dot(dZ2, self.W2.T)
    dP1 = dF1.reshape(self.P1.shape)
    dA1 = np.zeros_like(self.A1)

    for b in range(dP1.shape[0]):
        for h in range(dP1.shape[1]):
            for w in range(dP1.shape[2]):
                h_start = h * self.pool_size[0]
                w_start = w * self.pool_size[1]
                h_end = h_start + self.pool_size[0]
                w_end = w_start + self.pool_size[1]

                for c in range(dP1.shape[3]):
                    window = self.A1[b, h_start:h_end, w_start:w_end, c]
                    mask = np.zeros_like(window)
                    max_pos = np.unravel_index(window.argmax(), window.shape)
                    mask[max_pos] = 1
                    dA1[b, h_start:h_end, w_start:w_end, c] += mask * dP1[b, h, w, c]

    dZ1 = dA1 * relu_derivative(self.A1)
    dW1 = np.zeros_like(self.W1)

    for b in range(X.shape[0]):
        for h in range(self.Z1.shape[1] - self.kernel_size[0] + 1):
            for w in range(self.Z1.shape[2] - self.kernel_size[1] + 1):
                for f in range(self.W1.shape[3]):
                    patch = X[b, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :]
                    for c in range(X.shape[3]):
                        dW1[:, :, c, f] += patch[:, :, c] * dZ1[b, h, w, f]

    dB1 = np.sum(dZ1, axis=(0, 1, 2)) / m

    # Update
    self.W1 -= learning_rate * dW1
    self.B1 -= learning_rate * dB1
    self.W2 -= learning_rate * dW2
    self.B2 -= learning_rate * dB2
    self.W3 -= learning_rate * dW3
    self.B3 -= learning_rate * dB3

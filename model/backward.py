# import numpy as np
#
#
#
#
# def backwardPropagation(x, y, params, chache, leartning_rate):
#     '''
#     :param x:
#     :param y:
#     :param params:
#     :param chache:
#     :param leartning_rate:
#     :return:
#     '''
#
#
#     '''
#     apa kegunaan dari m??
#     '''
#     m = x.shape[0]
#
#
#
# '''
# Apa Kegunaan dari One_hot_encode??
# '''
# def One_hot_encode(y, m, output_size):
#
#     '''
#     :param y:
#     :param m:
#     :param output_size:
#     :return:
#     '''
#
#     if len(y.shape) == 1:
#         one_hot_y = np.zeros((m, output_size))
#         one_hot_y[np.arange(m), y] = 1
#     else:
#         one_hot_y = y
#     return one_hot_y
#
# def compute_output_layer_grads(A3, one_hot_y, A2, m):
#
#     '''
#     :param A3:
#     :param one_hot_y:
#     :param A2:
#     :param m:
#     :return:
#     '''
#
#     dZ3 = A3 - one_hot_y
#     dW3 = np.dot(A2.T, dZ3) / m
#     dB3 = np.sum(dZ3, axis=0) / m
#     return dZ3, dW3, dB3
#
# def copmute_fc_layer_grads(dZ3, W3, A2, F1, m):
#
#     '''
#     :param dZ3:
#     :param W3:
#     :param A2:
#     :param F1:
#     :param m:
#     :return:
#     '''
#
#     dZ2 = np.dot(dZ3, W3.T) * (A2 > 0)
#     dW2 = np.dot(F1.T, dZ2) / m
#     dB2 = np.sum(dZ2, axis=0) / m
#     return dZ2, dW2, dB2
#
# def compute_flatten_to_pooling_grads(dZ2, W2, P1_shape):
#
#     '''
#     :param dZ2:
#     :param W2:
#     :param P1_shape:
#     :return:
#     '''
#
#     dF1 = np.dot(dZ2, W2.T)
#     return dF1.reshape(P1_shape)
#
# def compute_pooling_to_conv_grads(dP1, A1, pool_size):
#
#     '''
#     :param dP1:
#     :param A1:
#     :param pool_size:
#     :return:
#     '''
#
#     dA1 = np.zeros_like(A1)
#     for b in range(dP1.shape[0]):
#         for h in range(dP1.shape[1]):
#             for w in range(dP1.shape[2]):
#                 h_start = h * pool_size[0]
#                 w_start = w * pool_size[1]
#                 h_end = h_start + pool_size[0]
#                 w_end = w_start + pool_size[1]
#
#                 for c  in range(dP1.shape[3]):
#                     window = A1[b, h_start:h_end, w_start:w_end, c]
#                     mask = np.zeros_like(window)
#                     max_pos = np.unravel_index(window.argmax(), window.shape)
#                     mask[max_pos]  = 1
#                     dA1[b, h_start:h_end, w_start:w_end, c] += dP1[b, h, w, c] * mask
#     return dA1
#
# def relu_backward(dA1, A1):
#
#     '''
#     :param dA1:
#     :param A1:
#     :return:
#     '''
#
#     return dA1 * (A1 > 0)
#
# def compute_conv_layer_grads(X, dZ1, kernel_size, W1_shape, Z1_shape):
#
#     '''
#     :param X:
#     :param dZ1:
#     :param kernel_size:
#     :param W1_shape:
#     :param Z1_shape:
#     :return:
#     '''
#
#     dW1 = np.zeros(W1_shape)
#     for b in range(X.shape[0]):
#         for h in range(Z1_shape[1] - kernel_size[0] + 1):
#             for w in range(Z1_shape[2] - kernel_size[1] + 1):
#                 for f in range(W1_shape[3]):
#                     patch = X[b, h:h + kernel_size[0], w:w + kernel_size[1], :]
#                     for c in range(X.shape[3]):
#                         dW1[:, :, c, f] += patch[:, :, c] * dZ1[b, h, w, f]
#     dB1 = np.sum(dZ1, axis=(0, 1, 2)) / X.shape[0]
#     return dW1, dB1
#
# def update_parameters(params, dW1, dB1, dW2, dB2, dW3, dB3, learning_rate):
#
#     '''
#     :param params:
#     :param dW1:
#     :param dB1:
#     :param dW2:
#     :param dB2:
#     :param dW3:
#     :param dB3:
#     :param learning_rate:
#     :return:
#     '''
#
#     params["W1"] -= learning_rate * dW1
#     params["B1"] -= learning_rate * dB1
#     params["W2"] -= learning_rate * dW2
#     params["B2"] -= learning_rate * dB2
#     params["W3"] -= learning_rate * dW3
#     params["B3"] -= learning_rate * dB3
#



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

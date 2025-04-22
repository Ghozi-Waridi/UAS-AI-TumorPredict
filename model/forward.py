import numpy as np
from .conv import conv2D
from .pooling import max_pooling
from .activations import relu, softmax
from .flatten import flatten


import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


'''
Proses Forward adalah proses untuk melakukan predict di dalam sebuah NeuralNetwroks.
proses ini dilakukan dengan cara menghitung dari inputlayer ke outputLayer
'''
def forward_pass(self, X, kernel=None):
    '''
    param: X: data yang ingin di prediksi
    param: kernel: ukuran kernel
    '''
    # if kernel is None:
    #     kernel = self.W1

    '''
    Jika gambar hanya 1, atau panjang shape 3 (tidak batch), maka akan di proses
    secara tunggal, atau batch tunggal, di expand di bagian depan
    '''
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)

    '''
    CONVULATION LAYER
    ------------------------------
    Jika batchnya hanya 1, maka gamba rakan di proses langsung,
    tapi jika batch gambar lebih dari 1 mka akan di lakukan perulangan 
    untuk memanggil kembali, fungsi conv2D, dengan input pergambar, dan memprosesnya
    dan memenuhi kondisi di atas
    '''
    if X.shape[0] == 1:
        self.Z1 = conv2D(X[0], kernel=kernel)
        self.Z1 = np.expand_dims(self.Z1, axis=0)
    else:
        self.Z1 = np.array([conv2D(x, kernel=kernel) for x in X])


    '''
    FUNGSI AKTIVASI, RELU
    '''
    self.A1 = relu(self.Z1)


    '''
    MAXPOOLING LAYER
    ------------------------------
    Jika batchnya hanya 1, maka gambar akan di proses langsung,
    tapi jika batch gambar lebih dari 1 mka akan di lakukan perulangan 
    untuk memanggil kembali, fungsi max_pooling dengan input pergambar, dan memprosesnya
    dan memenuhi kondisi di atas
    '''

    if self.A1.shape[0] == 1:
        self.P1 = max_pooling(self.A1[0], self.pool_size)
        self.P1 = np.expand_dims(self.P1, axis=0)
    else:
        self.P1 = np.array([max_pooling(a, self.pool_size) for a in self.A1])
    
    '''
    FLATTEN LAYER
    ------------------------------
    Fungsi ini digunakan untuk merubah dari matriks menjadi vector
    '''
    self.F1 = flatten(self.P1)
    
    '''
    FULLY CONNECTED LAYER
    -------------------------------
    Fungsi ini digunakan untuk menghubungkan antara layer satu dengan lainnya
    '''
    self.Z2 = np.dot(self.F1, self.W2) + self.B2
    self.A2 = relu(self.Z2)

    self.Z3 = np.dot(self.A2, self.W3) + self.B3
    self.A3 = softmax(self.Z3)

    return self.A3


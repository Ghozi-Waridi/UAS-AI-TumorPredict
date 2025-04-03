# import numpy as np
# from anyio.abc import value
# from hidden import  ReLu
#
# #####################################################
# ################ Forward Operations #################
# #####################################################
#
#
# def conv2D(images, kernel, stride=(1,1), padding=0):
#     '''
#         :param images: Gambar Input yang akan melakukan convulation
#         :param kernel:  Matirks untuk mengambil feature dari gambar tersebut (Gausjordan, sobel, edgeDetection, DLL)
#         :param stride: Matriks yang digunakan untuk melakukan perjalanan/celah
#         :param padding:  menambahkan pixel baru di dalam gambar supaya bisa melakukan convulation
#         :return: gambar yang sudah di lakukan kkonvulasi
#     '''
#
#     '''
#     Jika code yang digunakan menggunakan batch, dan bagian ini digunakan untuk mengubah setiap batch menjadi proses satu persatu
#     '''
#     if len(images.shape) == 4:
#         batch_size = images.shape[0]
#         results = []
#
#         for i in range(batch_size):
#             result = conv2D(images[i], kernel, stride, padding)
#             results.append(result)
#         return np.array(results)
#
#     image_np = np.array(images)
#
#     if len(image_np.shape) != 3:
#         raise ValueError("Gambar harus memiliki 3 dimensi height, width, channels")
#
#     if padding > 0:
#         image_pad = np.pad(image_np, ((padding, padding), (padding, padding), (0,0)), mode='constant')
#     else:
#         image_pad = image_np
#
#
#     '''
#     Mengambil Semua ukuran dari gambar, dan dimensinya
#     '''
#     h, w, c = image_pad.shape
#     kh, kw = kernel.shape[:2]
#
#     '''
#     Menghitung ukuran ouput atai dimensi dari outputnya,
#
#     kenapa kita harus mengukur output??
#     '''
#     out_h = (h - kh) // stride[0] + 1
#     out_w = (w - kw) // stride[0] + 1
#
#     out_c = kernel.shape[3] if len(kernel.shape) == 4 else 1
#     output = np.zeros((out_h, out_w, out_c))
#
#
#     '''
#     Convulation Operation
#     '''
#     for i in range(0, out_h):
#         for j in range(0, out_w):
#             '''
#             Mengapa dna dijadikan apa menghitung nilai i_pos, j_pos??
#             '''
#             i_pos = i * stride[0]
#             j_pos = j * stride[1]
#
#             '''
#             apa itu patch???
#             '''
#             patch = image_pad[
#                 i_pos:i_pos + kh, j_pos:j_pos + kw, :
#             ]
#
#             '''
#             Dibagian ini melakukan perhitungan antara kernel dan image, jelaskan proses??
#             '''
#             for f in range(out_c):
#                 for c_in in range(c):
#                     output[i, j, f] += np.sum(patch[:,:,c_in] * kernel[:,:,c_in, f])
#     return output
#
#
#
# def maxPooling(input_data, pool_size=(2,2), stride=None):
#     '''
#
#     :param input_data:
#     :param pool_size:
#     :param stride:
#     :return:
#     '''
#     if stride is None:
#         stride = pool_size
#
#     '''
#     Pengkondisian ini digunakan jika input data berupa
#     batch dari beberapa kumpulan gambar, jadi akan di ambil
#     untuk bagian gambarnya saja
#     '''
#     if len(input_data.shape) == 4:
#         batch_size = input_data.shape[0]
#
#         results = []
#         for i in range(batch_size):
#             result = maxPooling(input_data[i], pool_size, stride)
#             results.append(result)
#         return np.array(results)
#     input_height, input_width, input_channels = input_data.shape
#     pool_height, pool_width = pool_size
#     stride_height, stride_width = stride
#
#     output_height = (input_height - pool_height) // stride_height + 1 ### => Kenapa harus di tambah 1
#     output_width = (input_width - pool_width) // stride_width + 1
#
#     pooled_output = np.zeros((output_height, output_width, input_channels))
#
#
#     '''
#     Memulai Operations maxPooling
#     '''
#     for h in range(output_height):
#         for w in range(output_width):
#             '''
#             Menentukan posisi awal dari proses MaxPooling
#             '''
#             h_start = h * stride_height
#             w_start = w * stride_width
#
#             '''
#             Menentukan Akhir dari proses MaxPooling,
#             Dan kenapa harus menentukan nilai awal dan akhirnya ???
#             '''
#             h_end = h_start + pool_height
#             w_end = w_end + pool_width
#
#             for c in range(input_channels):
#                 patch = input_data[h_start:h_end, w_start:w_end, c]
#                 '''
#                 Kenapa pada bagian ini index yang digunakan
#                 untuk pooled_output harus h,w,c yang merupakan
#                 variabel dari for
#                 '''
#                 pooled_output[h,w,c] = np.max(patch)
#
#     return pooled_output
#
# '''
# Flatten : Proses merubah dari matriks menjadi vector
# '''
# def flatten(x):
#     '''
#
#     :param x:
#     :return:
#     '''
#
#     '''
#     Perhitungan flatten jika menggunakan bath
#     '''
#     if len(x.shape) == 4:
#         batch_size = x.shape(0)
#         return x.reshape(batch_size, -1)
#     else:
#         return x.shape(1,-1)
#
#
# def feedForward(x, W1, Z1, kernel=None):
#     if kernel is None:
#         kernel = W1
#
#     if len(x.shape) == 3:
#         x = np.expand_dims(x, axis=0)
#
#     if x.shape[0] == 1:
#         Z1 = conv2D(x[0], kernel=kernel)
#         Z1 = np.expand_dims(Z1, axis=0)
#     else:
#         Z1 = np.array([conv2D(x, kernel=kernel) for x in x])
#
#     A1 = ReLu(Z1)
#
#
#



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


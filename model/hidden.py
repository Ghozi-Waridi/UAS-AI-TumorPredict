# import numpy as np
# '''
# Semua Fungsi Aktivasi
# '''
#
# def ReLu(x):
#     '''
#
#     :param x:
#     :return:
#     '''
#     return np.maximum(0, x)
#
# def softMax(x):
#     '''
#     :param x:
#     :return:
#     '''
#     exps = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return exps/np.sum(exps, axis=1, keepdims=True)
#
# '''
# Menghitung loss
# '''
# def categoricalCrossEntropy(y_true, y_pred):
#     '''
#     :param y_true:
#     :param y_pred:
#     :return:
#     '''
#
#     if len(y_true.shape) == 1:
#         m = y_true.shape[0]
#         y_true_one_hot = np.zeros((m, y_pred.shape[1]))
#         y_true_one_hot[np.arange(m), y_true] = 1
#         y_true = y_true_one_hot
#
#     y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
#     loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
#     return loss

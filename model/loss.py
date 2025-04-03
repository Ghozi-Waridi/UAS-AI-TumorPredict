import numpy as np

'''
Melakukan perhitungan loss function
'''
def categorical_crossentropy(y_true, y_pred):
    if len(y_true.shape) == 1:
        m = y_true.shape[0]
        one_hot = np.zeros((m, y_pred.shape[1]))
        one_hot[np.arange(m), y_true] = 1
        y_true = one_hot

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

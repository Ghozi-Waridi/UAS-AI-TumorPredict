import numpy as np
import matplotlib.pyplot as plt
from ..model.NeuralNetworks import NeuralNetworks
from ..config.config import model_save_path, model_save_path4
from ..model.utils import save_model_params

import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def train_cnn_model(X_train, y_train, X_test, y_test, input_shape, num_classes, kernel, epochs=10, batch_size=32):
    '''
    params: X_train: training data
    params: y_train: training labels
    params: X_test: testing data
    params: y_test: testing labels
    params: input_shape: ukuran dari input data
    params: num_classes: jumlah dari class gambar (dataset), bentuk apa???
    params: kernel: kernel yang digunakan untuk training
    params: epochs: jumlah epoch untuk training
    params: batch_size: ukuran dari batch
    '''

    ### Apa itu Epochs = 
    ### Apa itu Batch Size
    model = NeuralNetworks(
        input_size=input_shape,
        hidden_size=32,
        output_size=num_classes,
        kernel_size=(3, 3),
        pool_size=(2, 2)
    )
    
    '''
    Melakukan training
    '''
    losses = model.train(X_train, y_train, kernel=kernel, epochs=epochs, batch_size=batch_size, learning_rate=0.01)
    model.save_model_params(model_save_path4)
    
    '''
    Mneghitung akurasi dari model
    '''
    accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    '''
    Melakukan plot hasil loos training
    '''
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return model

# import os
from model.NeuralNetworks import NeuralNetworks
from model.utils import load_model_params
from config.config import model_save_path
import numpy as np


import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Setup the model once and load weights
def get_model():
    input_shape = (64, 64, 1)
    model = NeuralNetworks(input_size=input_shape, hidden_size=32, output_size=2)
    model.load_model_params(model_save_path)
    return model

model = get_model()

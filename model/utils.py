import os
import pickle


'''
Melakukan penyimpanan dan pemuatan parameter model ke/dari file, file ini di panggil
setelah model seelsai di latih
'''
def save_model_params(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_data = {
        'input_size': self.input_size,
        'hidden_size': self.hidden_size,
        'output_size': self.output_size,
        'W1': self.W1,
        'W2': self.W2,
        'W3': self.W3,
        'B1': self.B1,
        'B2': self.B2,
        'B3': self.B3,
        'kernel_size': self.kernel_size,
        'pool_size': self.pool_size
    }
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model parameters saved to {path}")

def load_model_params(self, path):
    with open(path, 'rb') as f:
        model_data = pickle.load(f)

    self.input_size = model_data['input_size']
    self.hidden_size = model_data['hidden_size']
    self.output_size = model_data['output_size']
    self.W1 = model_data['W1']
    self.W2 = model_data['W2']
    self.W3 = model_data['W3']
    self.B1 = model_data['B1']
    self.B2 = model_data['B2']
    self.B3 = model_data['B3']
    self.kernel_size = model_data['kernel_size']
    self.pool_size = model_data['pool_size']

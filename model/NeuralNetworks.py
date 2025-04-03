import numpy as np
from .forward import forward_pass
from .backward import backward_pass
from .utils import save_model_params, load_model_params
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data.visualize import show_predictions


import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



class NeuralNetworks:
    '''
    Kita menentukan ukuran ukruan seperti pool, flatten, dll. 
    karena kita butuh tahu ukuran dimensi antar layer untuk membuat bobotnya.
    '''

    '''
    Bias : angka tambahan di setiap neuron yang membuat model lebih fleksibel. 
    weight : angka-angka yang menentukan seberapa besar pengaruh input terhadap output di tiap neuron.
    '''
    def __init__(self, input_size, hidden_size, output_size, kernel_size=(3, 3), pool_size=(2, 2)):
        '''
        params: input_size: ukuran dari input (height, width, channel)
        params: hidden_size: ukuran dari hidden layer
        params: output_size: ukuran dari output layer
        params: kernel_size: ukuran dari kernel (height, width)
        params: pool_size: ukuran dari pooling (height, width)
        '''

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        
        '''
        Pembuatan Nilai bobot awal untuk convulation layer
        bentuk : (tinggi_kernel, lebar_kernel, channel_input, jumlah_filter)
        Aku isi nilainya secara acak dan kecil (dikali 0.01), supaya saat 
        training nanti tidak terlalu besar
        '''
        self.W1 = np.random.randn(kernel_size[0], kernel_size[1], input_size[2], hidden_size) * 0.01
        
        '''
        Ini menghitung ukuran output setelah valid convolution (tanpa padding).
        '''
        conv_output_height = (input_size[0] - kernel_size[0] + 1)
        conv_output_width = (input_size[1] - kernel_size[1] + 1)

        '''
        Ini menghitung output setelah max pooling.
        '''
        pool_output_height = conv_output_height // pool_size[0]
        pool_output_width = conv_output_width // pool_size[1]

        '''
        Total neuron setelah hasil pooling dikonversi jadi 1 dimensi.
        '''
        flatten_size = pool_output_height * pool_output_width * hidden_size

        self.W2 = np.random.randn(flatten_size, hidden_size) * 0.01
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01

        self.B1 = np.zeros(hidden_size)
        self.B2 = np.zeros(hidden_size)
        self.B3 = np.zeros(output_size)

    def feedForward(self, X, kernel=None):
        return forward_pass(self, X, kernel)

    def backPropagation(self, X, Y, learning_rate):
        backward_pass(self, X, Y, learning_rate)

    
    '''
    Fungsi utama dari train model akan belajar dari data X dan label Y, selama beberapa epoch
    '''
    def train(self, X, Y, kernel=None, epochs=10, learning_rate=0.1, batch_size=32):
        print("Training model...", X.shape)
        '''
        params: X: data latih
        params: Y: label latih
        params: kernel: ukuran kernel
        params: epochs: jumlah epoch
        params: learning_rate: seberapa besar langkah perbaikan bobot.
        params: batch_size: ukuran batch 
        '''
        ### jumlah Sample
        m = X.shape[0] 
        ### Array menyimpan loss
        losses = []

        for epoch in range(epochs):
            '''
            Mengacak data untuk memberikan pembelajar yang lebih baik, dan supaya model tidak belajar
            urutan Data yang sama setiap epochs
            '''
            permutation = np.random.permutation(m)
            X_shuf = X[permutation]
            Y_shuf = Y[permutation]
            epoch_losses = []

            '''
            Disinilah pembuatan batch data, kita akan memecah data menjadi beberapa batch
            ------------------------------
            Data akan di pecah kembali yang awalnya per batch sekarang per image untuk operasi
            yang lebih mudah
            '''
            for i in range(0, m, batch_size):
                '''
                Mengambil data perbatch, dan di olah perbatch
                '''
                end = min(i + batch_size, m)

                '''
                Data yang sudah di ambil perbatcnya
                '''
                X_batch = X_shuf[i:end]
                Y_batch = Y_shuf[i:end]

                '''
                Menghitung hasil prediksi dari model, dan disinilah bagian pentingnya.
                model melakukan perhitungan dari input ke output. 
                feedForward, merupakan perhitungan layer kedepan dan back perhitungan untuk 
                memperbaiki loss dan parameter
                '''
                y_pred = self.feedForward(X_batch, kernel)
                self.backPropagation(X_batch, Y_batch, learning_rate)




                '''
                Menghitunga loss dari hasil prediksi, loss ini digunakan untuk mengukur 
                seberapa baik model kita, 
                dan hasil loss di masukan ke dalam array epoch loss utuk melihat dan sebagia history loss
                '''
                from .loss import categorical_crossentropy
                loss = categorical_crossentropy(Y_batch, y_pred)
                epoch_losses.append(loss)
            '''
            Rata-Rata loss dari setiap batch, dan di masukan ke dalam array losses
            '''
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        return losses

    def predict(self, X, kernel=None):
        '''
        param: X: data yang ingin di prediksi
        param: kernel: ukuran kernel
        '''
        '''
        Melakukan load parameters, dan melakukan predict
        '''
        # load_model_params(self, path)
        predictions = self.feedForward(X, kernel)
        return np.argmax(predictions, axis=1)

    def evaluate(self, x, y, classes, kernel=None):

        y_pred = self.predict(x)

        acc = np.mean(y_pred == y)
        print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

        print("\n📊 Classification Report:")
        print(classification_report(y, y_pred, target_names=classes))

        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        show_predictions(x, y, y_pred, classes)

    def save_model_params(self, path):
        save_model_params(self, path)

    def load_model_params(self, path):
        load_model_params(self, path)

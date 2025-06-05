from sklearn.metrics import classification_report, confusion_matrix

from config.config import model_save_path3, image_size, dataset_path, model_save_path, model_save_path4
from data.dataset_loader import load_data
from data.preprocess import preprocess_data
from data.visualize import visualize_samples
from inference.predict import predict_single_image

from utils.helpers import create_gaussian_kernel

from model.NeuralNetworks import NeuralNetworks


import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), label_encoder = load_data(dataset_path, img_size=image_size)
    classes = label_encoder.classes_

    visualize_samples(X_train, y_train, classes)

    X_train, X_test = preprocess_data(X_train, X_test)
    channels = X_train.shape[3] if len(X_train.shape) == 4 else 1
    input_shape = (image_size[0], image_size[1], channels)

    kernel = create_gaussian_kernel(output_channels=32, input_channels=channels)

    model = NeuralNetworks(
        input_size=input_shape,
        hidden_size=32,
        output_size=len(classes)
    )
    model.load_model_params(model_save_path)
    '''
    Tinggal mengubah parameter di bagian evalute file testing eval
    '''

    model.train(X_train, y_train, kernel=kernel, epochs=10, batch_size=32)
    model.evaluate(x=X_test, y=y_test, classes=classes)

    import os
    sample_image_path = os.path.join(dataset_path, classes[0], os.listdir(os.path.join(dataset_path, classes[0]))[0])
    print(sample_image_path)

    pred = predict_single_image(model, sample_image_path, image_size=image_size, label_encoder=label_encoder)
    print(f"Prediction result: {pred}")

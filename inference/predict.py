import cv2
import numpy as np

from utils.helpers import create_gaussian_kernel


def predict_single_image(model, image_path, image_size=(64, 64), label_encoder=None):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    channels = image.shape[3] if len(image.shape) == 4 else 1

    kernel = create_gaussian_kernel(output_channels=32, input_channels=channels)

    prediction = model.predict(image, kernel)
    if label_encoder:
        return label_encoder.inverse_transform(prediction)[0]
    return prediction[0]

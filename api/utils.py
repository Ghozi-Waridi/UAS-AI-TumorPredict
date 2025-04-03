import cv2
import numpy as np

def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

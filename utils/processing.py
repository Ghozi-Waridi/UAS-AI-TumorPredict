import os
import sys
from random import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def load_data(dataset_path, image_size=(64, 64), test_size=0.2, random_state=42):

    '''
    params: dataset_path: path atau jalur dari dataset (folder)
    params: image_size: ukuran untuk resize gambar supaya semuanya sama
    params: test_size: ukuran atau perbandingan seberapa banyak dataset yang akan digunakan utnuk testing (dibutuhkan skelarn)
    params: random_state: random state untuk membagi dataset menjadi training dan testing (dibutuhkan skelarn)
    '''

    images = []
    labels = []

    print("Loading Images...")

    class_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    print(f"Number of Classes: {len(class_folders)}, classes: {class_folders}")

    for class_name in class_folders:
        class_folder_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder_path):
            class_images = [
                f for f in os.listdir(class_folder_path) if f.lower().endswith(("jpg", "jpeg", "png"))
            ]

            print(f"Processing class {class_name} with {len(class_images)} images")

            for image_name in tqdm(class_images):
                image_path = os.path.join(class_folder_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = cv2.resize(image, image_size)
                        image = np.expand_dims(image, axis=-1)
                        images.append(image)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Error Processing Image: {image_path}")

    x = np.array(images)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.array(labels))

    print(f"Preprocessing complete. Images shape: {x.shape}, Labels shape: {y.shape}")

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test, label_encoder


def preprocess_data(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return X_train, X_test

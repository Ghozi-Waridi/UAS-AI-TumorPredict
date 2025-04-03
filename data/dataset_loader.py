import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


'''
Bagian ini kita mengmbil data data dari dataset folder, 
'''
def load_data(dataset_path, img_size=(64, 64), test_size=0.2, random_state=42):
    '''
    Param: dataset_path: path ke folder dataset
    Param: img_size: ukuran gambar yang diinginkan
    Param: test_size: proporsi data yang digunakan untuk testing (skalearn)
    Param: random_state: seed untuk random state (skalearn)
    '''
    images, labels = [], []

    class_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    print(f"Classes found: {class_folders}")

    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        for file in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, file)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, img_size)
                    image = np.expand_dims(image, axis=-1)
                    images.append(image)
                    labels.append(class_name)

    X = np.array(images)
    y = LabelEncoder().fit_transform(np.array(labels))

    return train_test_split(X, y, test_size=test_size, random_state=random_state), LabelEncoder().fit(labels)

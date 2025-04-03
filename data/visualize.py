import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_samples(X, y, classes, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        plt.subplot(1, num_samples, i + 1)
        if X[idx].shape[-1] == 1:
            plt.imshow(X[idx].squeeze(), cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Class: {classes[y[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_predictions(X, y_true, y_pred, class_names, num_samples=6):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].squeeze(), cmap="gray")
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"T: {true_label}\nP: {pred_label}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
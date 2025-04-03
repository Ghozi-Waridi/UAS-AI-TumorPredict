import numpy as np
import  cv2
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_samples(X, y, classes, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        plt.subplot(1, num_samples, i + 1)
        if X[idx].shape[-1] == 1:  # Grayscale
            plt.imshow(X[idx].squeeze(), cmap='gray')
        else:  
            plt.imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Class: {classes[y[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

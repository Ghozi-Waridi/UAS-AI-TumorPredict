# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from training.train import train_cnn_model
# from inference.predict import predict_single_image
# from sklearn.metrics import classification_report, confusion_matrix
# from config.config import *
# from data.dataset_loader import load_data
# from data.preprocess import preprocess_data
# from model.NeuralNetworks import NeuralNetworks
#
#
# def evaluate_model(x_test, y_test,):
#
#
#     # Predict
#     y_pred = model.predict(X_test)
#
#     # Accuracy
#     acc = np.mean(y_pred == y_test)
#     print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
#
#     # Classification Report
#     print("\n📊 Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=classes))
#
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.tight_layout()
#     plt.show()
#
#     # Show sample predictions (optional)
#     show_predictions(X_test, y_test, y_pred, classes)
#
#
# def show_predictions(X, y_true, y_pred, class_names, num_samples=6):
#     plt.figure(figsize=(15, 3))
#     for i in range(num_samples):
#         idx = np.random.randint(0, len(X))
#         plt.subplot(1, num_samples, i + 1)
#         plt.imshow(X[idx].squeeze(), cmap="gray")
#         true_label = class_names[y_true[idx]]
#         pred_label = class_names[y_pred[idx]]
#         color = "green" if true_label == pred_label else "red"
#         plt.title(f"T: {true_label}\nP: {pred_label}", color=color)
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()
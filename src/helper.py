import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Useful functions can be defined here so that they can be imported into other files
# -> better code organization and reusability



def load_data(dataset_path):
    data = np.load(dataset_path)

    X = data['X']
    y = data['y']

    return X, y

def format_data_for_pytorch(X, y):
    # reshape X to (num_samples, num_channels, num_time_steps)
    # because PyTorch expects (batch_size, num_channels, num_time_steps)
    X = np.transpose(X, (0, 2, 1))
    return X, y


# function to calculate the class weights for weighted cross entropy loss
def calculate_class_weights(y, scaling_factor=1.0, min_weight=0.05):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = 1.0 + (class_weights - 1.0) * scaling_factor
    class_weights = np.maximum(class_weights, min_weight)
    return torch.tensor(class_weights, dtype=torch.float32)

def plot_loss_history(train_loss_history, val_loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_features_with_pca(all_features, all_labels, n_components=2):
    
    pca = PCA(n_components=n_components)
    features_2d = pca.fit_transform(all_features)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    features_2d[:, 2],
    c=all_labels,
    cmap='viridis',
    alpha=0.7
)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA of Extracted Features')
    ax.legend(*scatter.legend_elements(), title="Classes")

    plt.show()



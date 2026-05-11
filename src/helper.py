import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
def calculate_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
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


# Code taken from https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate Binary Cross-Entropy Loss for each sample
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute pt (model confidence on true class)
        pt = torch.exp(-BCE_loss)
        
        # Apply the focal adjustment
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
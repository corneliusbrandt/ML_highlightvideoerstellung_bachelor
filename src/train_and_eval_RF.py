from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from helper import calculate_class_weights, load_data, plot_loss_history
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from focal_loss import FocalLoss
from model_architecures import CNN1D_V1, CNN1D_V2, CNN1D_V3
import matplotlib.pyplot as plt
import numpy as np


'''
This script is responsible for training and evaluating different model architectures with Random Forest as a classifier.
'''


def get_features_and_labels(model, dataloader): 
    '''
    This function runs the feature extractor part of the model and returns the 
    extracted features and the corresponding labels for the whole dataset
    '''
    
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            features = model.feature_extractor(X_batch)
            all_features.append(features.numpy())
            all_labels.append(y_batch.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_features = all_features.squeeze(-1)
    return all_features, all_labels


class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # reshape X to (num_windows, num_channels, time) from (num_windows, time, num_channels)
        # because PyTorch expects it
        self.X = self.X.permute(0, 2, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Data preparation
batch_size = 32
num_classes = 2


#Training Loop
n_epochs = 200
learning_rate = 0.00001
weight_scaling_factor = 2.5
min_weight = 1.5

# Early Stopping Parameters
early_stopping_patience = 20
best_val_loss = float('inf')
min_delta = 0.0001
epochs_without_improvement = 0
best_epoch = 0

# threshold for converting probabilities to binary predictions
threshold = 0.57

# saving the best model
best_model_path = r"src\Models\best_model_RF.pth"





# Import Data and prepare it for Training
X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_val, y_val = load_data("datasets_output/val_dataset_augmented_binary.npz")
val_dataset = Dataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Calculate class weights for imbalanced dataset
class_weights = calculate_class_weights(y_train, num_classes=num_classes, scaling_factor=weight_scaling_factor, min_weight=min_weight)
print(f"Class Weights: {class_weights}")

# Initialize Model, Loss Function and Optimizer
model = CNN1D_V3(num_channels=27, num_classes=num_classes)

rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
    )
#loss_function = nn.CrossEntropyLoss(weight=class_weights)
loss_function = FocalLoss(gamma=3, alpha=class_weights, task_type='multi-class', num_classes=num_classes)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Containers for average metric calculation and history tracking
train_loss_history = []
val_loss_history = []



# Training and Evaluation Loop
for epoch in range(n_epochs):
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0
    val_loss = 0
    val_correct = 0
    val_total = 0

    for X_batch, y_batch in train_loader:
        pred_logit = model(X_batch)
        loss = loss_function(pred_logit, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.argmax(pred_logit, dim=1)
        train_correct += (pred == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_accuracy = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)


    # Train Random Forest on extracted features
    X_train_features, y_train_labels = get_features_and_labels(model, train_loader)
    X_val_features, y_val_labels = get_features_and_labels(model, val_loader)

    rf.fit(X_train_features, y_train_labels)
    preds = rf.predict(X_val_features)
    # Warning: the loss here is just so the early stopping can work, it is not the same as the one used
    # for the training model and the two should not be compared
    val_loss = log_loss(y_val_labels, rf.predict_proba(X_val_features))
    #val_loss = loss_function(torch.tensor(preds), torch.tensor(y_val_labels)).item()
    

    # Calculate metrics and print results
    avg_val_loss = val_loss / len(val_loader)
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    print(classification_report(y_val_labels, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val_labels, preds))


    # Early stopping
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)  # Save the best model
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch} epochs.")
        print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
        break

    

plot_loss_history(train_loss_history, val_loss_history)

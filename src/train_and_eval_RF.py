from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from helper import calculate_class_weights, load_data, plot_loss_history
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from focal_loss import FocalLoss
from model_architecures import CNN1D_V1, CNN1D_V2, CNN1D_V3
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score)
from helper import save_and_load_pickle
from torchinfo import summary


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
num_channels = 27


#Training Loop
n_epochs = 200
learning_rate = 0.00001
weight_scaling_factor = 2
min_weight = 0.01
gamma = 3

# Early Stopping Parameters
early_stopping_patience = 20
best_val_loss = float('inf')
best_f1_score = 0.0
min_delta = 0.0001
epochs_without_improvement = 0
best_epoch = 0
early_stopping_monitor = 'f1'  # Options: 'val_loss' or 'f1'

# saving the best model
best_cnn_path = r"src\Models\CNN_RF\CNN_RF.pth"
best_rf_path = r"src\Models\CNN_RF\RF.pkl"


#---------------------------------------------------------------------------
# Import Data and prepare it for Training
#---------------------------------------------------------------------------
X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
X_val, y_val = load_data("datasets_output/val_dataset_augmented_binary.npz")
val_dataset = Dataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#--------------------------------------------------------------------------
# Calculate class weights for imbalanced dataset
#--------------------------------------------------------------------------
class_weights = calculate_class_weights(y_train, num_classes=num_classes, scaling_factor=weight_scaling_factor, min_weight=min_weight)
print(f"Class Weights: {class_weights}")

#--------------------------------------------------------------------------
# Initialize Model, Loss Function and Optimizer
#--------------------------------------------------------------------------
model = CNN1D_V3(num_channels=num_channels, num_classes=num_classes)
rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
    )
loss_function = FocalLoss(gamma=gamma, alpha=class_weights, task_type='multi-class', num_classes=num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


#-------------------------------------------------------------------------
# Containers for average metric calculation and history tracking
#-------------------------------------------------------------------------
train_loss_history = []
val_loss_history = []
val_precision_history = []
val_recall_history = []
val_f1_history = []


#-------------------------------------------------------------------------
# Training Loop
#-------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------------
    # Validation
    # Warning: the loss here is just so the early stopping can work, it is not the same as the one used
    # for the training model and the two should not be compared
    #---------------------------------------------------------------------------------------
    preds = rf.predict(X_val_features)
    val_loss = log_loss(y_val_labels, rf.predict_proba(X_val_features))
    
    #-------------------------------------------------------------------------------------
    # Calculate metrics
    #-------------------------------------------------------------------------------------
    avg_val_loss = val_loss
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    val_accuracy = accuracy_score(y_val_labels, preds)
    val_precision = precision_score(y_val_labels, preds)
    val_precision_history.append(val_precision)
    val_recall = recall_score(y_val_labels, preds)
    val_recall_history.append(val_recall)
    val_f1 = f1_score(y_val_labels, preds)
    val_f1_history.append(val_f1)
    val_confusion_matrix = confusion_matrix(y_val_labels, preds)

    #-------------------------------------------------------------------------------------
    # Early stopping
    #-------------------------------------------------------------------------------------
    if early_stopping_monitor == 'f1':
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            epochs_without_improvement = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_cnn_path)  # Save the best model
            save_and_load_pickle(rf, best_rf_path, 'save')  # Save the best Random Forest classifier
            
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            print(f"Best F1 Score: {best_f1_score:.4f} at Epoch {best_epoch}")
            break

    elif early_stopping_monitor == 'val_loss':
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_cnn_path)  # Save the best model
            save_and_load_pickle(rf, best_rf_path, 'save')  # Save the best Random Forest classifier
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
            break



#------------------------------------------------------------------------
# Print Epoch Summary
#------------------------------------------------------------------------
    print(
        f"Epoch {epoch+1}/{n_epochs}, "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Val Acc: {val_accuracy:.4f}, "
        f"Val F1: {val_f1:.4f},"
        f"\nVal Confusion Matrix:\n{val_confusion_matrix}"
    )



#-------------------------------------------------------------------------
# Final Metrics Summary
#-------------------------------------------------------------------------
print(
    f"Best Validation Precision: {val_precision_history[best_epoch - 1]:.4f}\n"
    f"Best Validation Recall: {val_recall_history[best_epoch - 1]:.4f}\n"
    f"Best Validation F1: {val_f1_history[best_epoch - 1]:.4f}"
)

summary(model, input_size=(batch_size, num_channels, 120))
plot_loss_history(train_loss_history, val_loss_history)

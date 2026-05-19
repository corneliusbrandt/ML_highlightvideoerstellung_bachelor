from helper import calculate_class_weights, load_data, plot_loss_history
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from focal_loss import FocalLoss

'''
Second draft of a 1D CNN for binary classification.
Changes from V1:
- Added Batch Normalization after each convolutional layer for better training stability and faster convergence.
- Added Dropout after each convolutional layer to reduce overfitting.
- Reduced the number of filters in the convolutional layers to prevent overfitting and reduce computational complexity.
- Smaller classifier
- Smaller Architecure overall to prevent overfitting.
'''

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
    
class CNN1D(nn.Module):
    def __init__(self, num_channels=27, num_classes=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=25, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=25, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    


num_classes = 2

# Import Data and prepare it for Training
X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_val, y_val = load_data("datasets_output/val_dataset_binary.npz")
val_dataset = Dataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# Calculate class weights for imbalanced dataset
class_weights = calculate_class_weights(y_train, scaling_factor=2.5, min_weight=1.5)

print(f"Class Weights: {class_weights}")

# Initialize Model, Loss Function and Optimizer
model = CNN1D(num_channels=27, num_classes=num_classes)
#loss_function = nn.CrossEntropyLoss(weight=class_weights)
loss_function = FocalLoss(gamma=3, alpha=class_weights, task_type='multi-class', num_classes=num_classes)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)


#Training Loop
n_epochs = 200
train_loss_history = []
val_loss_history = []

# Containers for average metric calculation
val_precision_history = []
val_recall_history = []
val_f1_history = []

# Early Stopping Parameters
early_stopping_patience = 30
best_val_loss = float('inf')
min_delta = 0.0001
epochs_without_improvement = 0
best_epoch = 0


for epoch in range(n_epochs):
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

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


    # Validation
    model.eval()

    with torch.no_grad():

        val_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        for X_batch, y_batch in val_loader:
            pred_logit = model(X_batch)
            loss = loss_function(pred_logit, y_batch)

            val_loss += loss.item()

            pred = torch.argmax(pred_logit, dim=1)

            all_preds.extend(pred.numpy())
            all_labels.extend(y_batch.numpy())
            val_correct += (pred == y_batch).sum().item()
            val_total += y_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)


    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)


    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_epoch = epoch + 1
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch} epochs.")
        print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
        break

    # Calculate metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds)
    val_precision_history.append(val_precision)
    val_recall = recall_score(all_labels, all_preds)
    val_recall_history.append(val_recall)
    val_f1 = f1_score(all_labels, all_preds)
    val_f1_history.append(val_f1)
    val_confusion_matrix = confusion_matrix(all_labels, all_preds)

    print(
        f"Epoch {epoch+1}/{n_epochs}, "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Val Acc: {val_accuracy:.4f}, "
        f"Val F1: {val_f1:.4f},"
        f"\nVal Confusion Matrix:\n{val_confusion_matrix}"
    )

avg_val_precision = sum(val_precision_history) / len(val_precision_history)
avg_val_recall = sum(val_recall_history) / len(val_recall_history)
avg_val_f1 = sum(val_f1_history) / len(val_f1_history)

print(
    f"Average Validation Precision: {avg_val_precision:.4f}\n"
    f"Best Validation Precision: {val_precision_history[best_epoch - 1]:.4f}\n"
    f"Average Validation Recall: {avg_val_recall:.4f}\n"
    f"Best Validation Recall: {val_recall_history[best_epoch - 1]:.4f}\n"
    f"Average Validation F1: {avg_val_f1:.4f}\n"
    f"Best Validation F1: {val_f1_history[best_epoch - 1]:.4f}"
)

plot_loss_history(train_loss_history, val_loss_history)

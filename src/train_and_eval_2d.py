from helper import calculate_class_weights, load_data, plot_loss_history
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from focal_loss import FocalLoss
from model_architecures import CNN1D_V1, CNN1D_V2
from torchvision.models import resnet18


'''
This script is responsible for training and evaluating different model architectures.
The modifiable parameters are:
    - Model Architecture (needs to be adjusted further down in the code)
    - num_classes
    - n_epochs
    - learning_rate
    - weight_scaling_factor
    - min_weight
    - early_stopping_patience
    - min_delta
    - threshold for converting probabilities to binary predictions
'''


# Datapreparation
batch_size = 32
num_classes = 2


#Training Loop
n_epochs = 200
learning_rate = 0.0001
weight_scaling_factor = 4
min_weight = 0.5

# Early Stopping Parameters
early_stopping_patience = 30
best_val_loss = float('inf')
min_delta = 0.0001
epochs_without_improvement = 0
best_epoch = 0

# threshold for converting probabilities to binary predictions
threshold = 0.57

# saving the best model
best_model_path = r"src\Models\best_model_ResNet.pth"




class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # Reshape to (num_windows, num_channels, height, width) for 2D CNN input
        self.X = self.X.unsqueeze(1)
        # Repeat the single channel to create a 3-channel input for the CNN
        self.X = self.X.repeat(1, 3, 1, 1)  
        

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


# Import Data and prepare it for Training
X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_val, y_val = load_data("datasets_output/val_dataset_binary.npz")
val_dataset = Dataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Calculate class weights for imbalanced dataset
class_weights = calculate_class_weights(y_train, scaling_factor=weight_scaling_factor, min_weight=min_weight)
print(f"Class Weights: {class_weights}")

# Initialize Model, Loss Function and Optimizer
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer for binary classification
#loss_function = nn.CrossEntropyLoss(weight=class_weights)
loss_function = FocalLoss(gamma=3.5, alpha=class_weights, task_type='multi-class', num_classes=num_classes)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Containers for average metric calculation and history tracking
val_precision_history = []
val_recall_history = []
val_f1_history = []
train_loss_history = []
val_loss_history = []


# Training and Evaluation Loop
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


    # Validation, In validation mode a softmax funcion is applied to make use of a threshold for binary classification
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

            probs = torch.softmax(pred_logit, dim=1)

            pred = (probs[:, 1] > threshold).long()  # Convert probabilities to binary predictions

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
        torch.save(model.state_dict(), best_model_path)  # Save the best model
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
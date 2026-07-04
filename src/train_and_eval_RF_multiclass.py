from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( classification_report, confusion_matrix, log_loss, accuracy_score, precision_score, recall_score, f1_score)
from src.helper import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.focal_loss import FocalLoss
from src.model_architecures import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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


def run_train_and_eval_RF_multiclass(debug=False):
    print("Running train_and_eval_RF_multiclass...")
    # Data preparation
    batch_size = 32
    num_classes = 6
    num_channels = 27

    #Training Loop
    n_epochs = 2000
    learning_rate = 0.00001
    weight_scaling_factor = 3
    min_weight = 0.01
    gamma = 3

    # Early Stopping Parameters
    early_stopping_patience = 20
    best_val_loss = float('inf')
    best_f1_score = 0.0
    min_delta = 0.0001
    epochs_without_improvement = 0
    best_epoch = 0
    early_stopping_monitor = 'f1'  # Can be 'val_loss' or 'f1'

    # saving the best model
    best_cnn_path = r"src\Models\CNN_RF\CNN_RF_multiclass.pth"
    best_rf_path = r"src\Models\CNN_RF\RF_multiclass.pkl"


    #----------------------------------------------------------------------------
    # Import Data and prepare it for Training
    #----------------------------------------------------------------------------
    X_train, y_train = load_data("datasets_output/train_dataset_augmented.npz")
    train_dataset = Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_val, y_val = load_data("datasets_output/val_dataset_augmented.npz")
    val_dataset = Dataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #----------------------------------------------------------------------------
    # Calculate class weights for imbalanced dataset
    #----------------------------------------------------------------------------
    class_weights = calculate_class_weights(y_train, num_classes=num_classes, scaling_factor=weight_scaling_factor, min_weight=min_weight)
    print(f"Class Weights: {class_weights}")

    #----------------------------------------------------------------------------
    # Initialize Model, Loss Function and Optimizer
    #----------------------------------------------------------------------------
    model = CNN1D_V3(num_channels=num_channels, num_classes=num_classes)

    rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
        )

    loss_function = FocalLoss(gamma=gamma, alpha=class_weights, task_type='multi-class', num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #----------------------------------------------------------------------------
    # Containers for average metric calculation and history tracking
    #----------------------------------------------------------------------------
    val_precision_history = []
    val_recall_history = []
    val_f1_history = []
    train_loss_history = []
    val_loss_history = []


    #----------------------------------------------------------------------------
    # Training Loop
    #----------------------------------------------------------------------------
    for epoch in range(n_epochs):
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
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

    
        X_train_features, y_train_labels = get_features_and_labels(model, train_loader)
        X_val_features, y_val_labels = get_features_and_labels(model, val_loader)
        rf.fit(X_train_features, y_train_labels)

        #----------------------------------------------------------------------------
        # Validation
        # Warning: the loss here is just so the early stopping can work, it is not the same as the one used
        # for the training model and the two should not be compared
        #----------------------------------------------------------------------------
        preds = rf.predict(X_val_features)
        proba = rf.predict_proba(X_val_features)

        proba_full = np.zeros((proba.shape[0], num_classes))

        for i, cls in enumerate(rf.classes_):
            proba_full[:, cls] = proba[:, i]

        val_loss = log_loss(
            y_val_labels,
            proba_full,
            labels=list(range(num_classes))
            )
        
        #----------------------------------------------------------------------------
        # Calculate metrics
        #----------------------------------------------------------------------------
        avg_val_loss = val_loss
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        val_accuracy = accuracy_score(y_val_labels, preds)
        val_precision = precision_score(y_val_labels, preds, average='macro', zero_division=0)
        val_precision_history.append(val_precision)
        val_recall = recall_score(y_val_labels, preds, average='macro', zero_division=0)
        val_recall_history.append(val_recall)
        val_f1 = f1_score(y_val_labels, preds, average='macro', zero_division=0)
        val_f1_history.append(val_f1)
        val_confusion_matrix = confusion_matrix(y_val_labels, preds, labels=list(range(num_classes)))

        #-------------------------------------------------------------------------
        # Early stopping
        #-------------------------------------------------------------------------
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
        if debug:
            print(
                f"Epoch {epoch+1}/{n_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}, "
                f"Val F1: {val_f1:.4f},"
                f"\nVal Confusion Matrix:\n{val_confusion_matrix}"
            )

            print(classification_report(y_val_labels, preds, labels=list(range(num_classes)), zero_division=0))


    #------------------------------------------------------------------------
    # Final Metrics Summary
    #------------------------------------------------------------------------    

    avg_val_precision = sum(val_precision_history) / len(val_precision_history)
    avg_val_recall = sum(val_recall_history) / len(val_recall_history)
    avg_val_f1 = sum(val_f1_history) / len(val_f1_history)


    print(
        #f"Average Validation Precision: {avg_val_precision:.4f}\n"
        f"Best Validation Precision: {val_precision_history[best_epoch - 1]:.4f}\n"
        #f"Average Validation Recall: {avg_val_recall:.4f}\n"
        f"Best Validation Recall: {val_recall_history[best_epoch - 1]:.4f}\n"
        #f"Average Validation F1: {avg_val_f1:.4f}\n"
        f"Best Validation F1: {val_f1_history[best_epoch - 1]:.4f}"
    )
    
    if debug:
        summary(model, input_size=(batch_size, num_channels, 120))
        plot_loss_history(train_loss_history, val_loss_history)

    print("------------")


if __name__ == "__main__":
    run_train_and_eval_RF_multiclass(debug=True)
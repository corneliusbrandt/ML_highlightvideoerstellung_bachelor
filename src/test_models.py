from helper import load_data, save_and_load_pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from model_architecures import CNN1D_V1, CNN1D_V2, CNN1D_V3
from torchinfo import summary
import numpy as np
#from train_and_eval_RF import get_features_and_labels

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


# Model mode (if True, the script will evaluate a combination of CNN and Random Forest)
rf_combination = True
class_mode = 'binary'  # 'binary' or 'multiclass'


# Datapreparation
batch_size = 32
num_classes = 2
num_channels = 27


cnn_model_path = r"src\Models\CNN_RF\CNN_RF.pth"
rf_model_path = r"src\Models\CNN_RF\RF.pkl"

#---------------------------
# Load test dataset
#---------------------------
X_test, y_test = load_data("datasets_output/test_dataset_binary.npz")
test_dataset = Dataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#--------------------------
# Load the trained model
#--------------------------
cnn_model = CNN1D_V3(num_channels=num_channels, num_classes=num_classes)
cnn_model.load_state_dict(torch.load(cnn_model_path))
rf_model = None

if rf_combination:
    rf_model = save_and_load_pickle(None, rf_model_path, mode="load")

#--------------------------
# Evaluate the model on the test dataset
#--------------------------
all_labels = []
all_preds = []

if rf_combination:
    print("Evaluating combined CNN and Random Forest model")

    X_test_features, y_test_labels = get_features_and_labels(cnn_model, test_loader)
    all_preds = rf_model.predict(X_test_features)
    all_labels = y_test_labels
else:
    cnn_model.eval()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred_logit = cnn_model(X_batch)

            probs = torch.softmax(pred_logit, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

#--------------------------
# Calculate metrics
#--------------------------
if class_mode == "multiclass":
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
elif class_mode == "binary":
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

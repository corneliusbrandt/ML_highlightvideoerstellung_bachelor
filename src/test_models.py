from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.helper import *
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from src.model_architecures import CNN1D_V1, CNN1D_V2, CNN1D_V3
from torchinfo import summary
import numpy as np
from tsai.all import *
from typing import Literal
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
    
ModelArchitecture = Literal['CNN1D_V3', 'ResNet', 'InceptionTime', 'FCN', 'OmniScaleCNN']


def run_test_models(rf_combination=False,
                    evaluation_mode='multiclass',
                    num_classes=6,
                    cnn_model_path=r"src\Models\CNN_multiclass.pth",
                    rf_model_path=r"src\Models\CNN_RF\RF_multiclass.pkl",
                    test_dataset_path="datasets_output/test_dataset.npz",
                    model_architecture: ModelArchitecture='CNN1D_V3',
                    debug=False):
    
    '''
    rf_combination: bool, If set to true, the function will evaluate a combination of CNN and Random Forest model.
    evaluation_mode: str, 'multiclass' or 'binary', determines the evaluation metrics
    num_classes: int, number of classes in the classification task
    cnn_model_path: str, path to the trained CNN model
    rf_model_path: str, path to the trained Random Forest model
    test_dataset_path: str, path to the test dataset
    model_architecture: str, the architecture of the model to be tested
    debug: bool, if set to True, additional debug information will be printed
    '''


    # Datapreparation
    batch_size = 32
    num_channels = 27

    #---------------------------
    # Load test dataset
    #---------------------------
    X_test, y_test = load_data(test_dataset_path)
    test_dataset = Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("X_test:", X_test.shape, X_test.dtype, X_test.flags["C_CONTIGUOUS"])
    print("y_test:", y_test.shape, y_test.dtype)
    print("Klassen Test:", np.unique(y_test, return_counts=True))


    #--------------------------
    # Load the trained model
    #--------------------------
    if model_architecture == 'CNN1D_V3':
        cnn_model = CNN1D_V3(num_channels=num_channels, num_classes=num_classes)
    elif model_architecture == 'ResNet':
        cnn_model = ResNet(
            c_in=num_channels,
            c_out=num_classes,
        )
    elif model_architecture == 'InceptionTime':
        cnn_model = InceptionTime(
            c_in=num_channels,
            c_out=num_classes,
        )
    elif model_architecture == 'FCN':
        cnn_model = FCN(
            c_in=num_channels,
            c_out=num_classes,
        )
    elif model_architecture == 'OmniScaleCNN':
        cnn_model = OmniScaleCNN(
            c_in=num_channels,
            c_out=num_classes,
            seq_len=X_test.shape[1]
        )
    else:
        raise ValueError(f"Invalid model architecture: {model_architecture}. Choose from 'CNN1D_V3', 'ResNet', 'InceptionTime', 'FCN', or 'OmniScaleCNN'.")
        return
        

    cnn_model.load_state_dict(torch.load(cnn_model_path))
    rf_model = None

    if rf_combination:
        rf_model = save_and_load_pickle(None, rf_model_path, mode="load")

    #--------------------------
    # Evaluate the model on the test dataset
    #--------------------------
    all_labels = []
    all_preds = []
    softmax_outputs = []

    if rf_combination:
        print("Evaluating combined CNN and Random Forest model")

        X_test_features, y_test_labels = get_features_and_labels(cnn_model, test_loader)
        all_preds = rf_model.predict(X_test_features)
        softmax_outputs = rf_model.predict_proba(X_test_features)
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
                softmax_outputs.extend(probs.numpy())

    #--------------------------
    # Calculate metrics
    #--------------------------
    if evaluation_mode == "multiclass":
        print("\nEvaluating multiclass classification metrics")
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

        # Convert labels and predictions to binary to be able to compare it to the binary evaluation metrics
        print("\nEvaluating multiclass metrics as binary classification metrics")
        binary_labels = np.where(np.array(all_labels) == 5, 0, 1)
        binary_preds = np.where(np.array(all_preds) == 5, 0, 1)

        accuracy_binary_calc = accuracy_score(binary_labels, binary_preds)
        precision_binary_calc = precision_score(binary_labels, binary_preds, average='binary', zero_division=0)
        recall_binary_calc = recall_score(binary_labels, binary_preds, average='binary', zero_division=0)
        f1_binary_calc = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)
        cm_binary_calc = confusion_matrix(binary_labels, binary_preds)

        print(f"Test Accuracy (binary): {accuracy_binary_calc:.4f}")
        print(f"Test Precision (binary): {precision_binary_calc:.4f}")
        print(f"Test Recall (binary): {recall_binary_calc:.4f}")
        print(f"Test F1-Score (binary): {f1_binary_calc:.4f}")
        print("Confusion Matrix (binary):")
        print(cm_binary_calc)

    elif evaluation_mode == "binary":
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

    if debug:
        summary(cnn_model, input_size=(batch_size, num_channels, 120))
        plot_onehot_predictions_with_slider(all_labels, all_preds, num_classes, window=200)
        plot_softmax_with_slider(softmax_outputs, y_true=all_labels, num_classes=num_classes, window=200)
        plot_confidence_with_slider(softmax_outputs, y_true=all_labels, y_pred=all_preds, window=200)

    print("------------")

    if evaluation_mode == "multiclass":
        return accuracy_binary_calc, precision_binary_calc, recall_binary_calc, f1_binary_calc, cm, all_labels, all_preds, softmax_outputs
    elif evaluation_mode == "binary":
        return accuracy, precision, recall, f1, cm, all_labels, all_preds, softmax_outputs



if __name__ == "__main__":
    run_test_models(rf_combination=False,
                    evaluation_mode='multiclass',
                    num_classes=6,
                    cnn_model_path=r"src\Models\CNN_multiclass.pth",
                    rf_model_path=r"src\Models\CNN_RF\RF_multiclass.pkl",
                    test_dataset_path="datasets_output/test_dataset.npz",
                    model_architecture='CNN1D_V3',
                    debug=True)
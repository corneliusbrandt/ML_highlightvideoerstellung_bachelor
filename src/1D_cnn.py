from helper import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    



X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")

train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

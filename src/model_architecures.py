import torch
import torch.nn as nn

class CNN1D_V1(nn.Module):
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
    


'''
Second draft of a 1D CNN for binary classification.
Changes from V1:
- Added Batch Normalization after each convolutional layer for better training stability and faster convergence.
- Added Dropout after each convolutional layer to reduce overfitting.
- Reduced the number of filters in the convolutional layers to prevent overfitting and reduce computational complexity.
- Smaller classifier
- Smaller Architecture overall to prevent overfitting.
'''

class CNN1D_V2(nn.Module):
    def __init__(self, num_channels=27, num_classes=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=25, padding=12),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),



            nn.Conv1d(16, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)


        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    

    
class CNN1D_V3(nn.Module):
    def __init__(self, num_channels=27, num_classes=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=25, padding=12),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),


            nn.Conv1d(16, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=2),


            nn.Conv1d(32, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)


        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
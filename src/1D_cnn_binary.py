from helper import calculate_class_weights, load_data, plot_loss_history
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
    



# Import Data and prepare it for Training
X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Calculate class weights for imbalanced dataset
class_weights = calculate_class_weights(y_train)

# Initialize Model, Loss Function and Optimizer
model = CNN1D(num_channels=27, num_classes=2)
loss_function = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#Training Loop
n_epochs = 50
loss_history = []

for epoch in range(n_epochs):
    model.train()

    total_loss = 0
    pred_correct = 0
    pred_total = 0

    for X_batch, y_batch in train_loader:
        pred_logit = model(X_batch)
        loss = loss_function(pred_logit, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(pred_logit, dim=1)
        pred_correct += (pred == y_batch).sum().item()
        pred_total += y_batch.size(0)

        accuracy = pred_correct / pred_total

    loss_history.append(total_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

plot_loss_history(loss_history)

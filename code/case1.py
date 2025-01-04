import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Paths to the labeled feature CSV files and label file
feature_dir = '../data/DNA_features'
feature_files = [
    'd_0_fraq_1865combined.csv',
    'd_1_fraq_1865combined.csv',
    'd_2_fraq_1865combined.csv',
    'd_3_fraq_1865combined.csv'
]
label_file = '../data/data1865_label.csv'
# output_representation_file = '../data/data_2/1865_representation_new.csv'

label_df = pd.read_csv(label_file, header=None, names=['File', 'Label'])

# Initialize a list to hold feature arrays for each channel
features = []

for feature_file in feature_files:
    df = pd.read_csv(os.path.join(feature_dir, feature_file), header=None)

    feature_array = df.iloc[:, 1:].values  # Shape: (1865, 4096)
    feature_array = feature_array.reshape(-1, 64, 64)  # Shape: (1865, 64, 64)
    features.append(feature_array)

X = np.stack(features, axis=1)  # Shape: (1865, 4, 64, 64)
y = label_df['Label'].values  # Shape: (1865,)

# =============================
# Step 1: Convert to PyTorch Tensors
# =============================

# Convert feature data to float32 tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


# =============================
# Step 2: Define the CNN Model
# =============================

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Input channels: 4, Output channels: 16
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)  # Output single value for binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (batch, 16, 64, 64)
        x = self.pool(x)  # (batch, 16, 32, 32)
        x = torch.relu(self.conv2(x))  # (batch, 32, 32, 32)
        x = self.pool(x)  # (batch, 32, 16, 16)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 32*16*16)
        x = torch.relu(self.fc1(x))  # (batch, 64)
        x = torch.sigmoid(self.fc2(x))  # (batch, 1)
        return x


# =============================
# Step 3: Train the Final Model on All Data
# =============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ConvNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

num_epochs = 50
best_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Scheduler step
    scheduler.step(epoch_loss)

    # Early Stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        trigger_times = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        print(f"EarlyStopping counter: {trigger_times} out of {patience}")
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# =============================
# Step 4: Load and Preprocess New Unlabeled Data
# =============================

# Paths to the new feature CSV files
# new_feature_files = [
#     'contigs_D3330_d0.csv',
#     'contigs_D3330_d1.csv',
#     'contigs_D3330_d2.csv',
#     'contigs_D3330_d3.csv'
# ]
# new_data_dir = '../data/case/casestudy/Month0'  # Replace with your actual new data directory
# new_output_file = '../data/case/casestudy/Month0/predictions/Month0_D3330_predictions_2.csv'  # Path to save predictions

new_feature_files = [
    'crassphage_d0.csv',
    'crassphage_d1.csv',
    'crassphage_d2.csv',
    'crassphage_d3.csv'
]
new_data_dir = '../data/case'
new_output_file = '../data/case/predictions/case1_predictions.csv'
# Read the new feature CSV files
new_features = []

for feature_file in new_feature_files:
    df = pd.read_csv(os.path.join(new_data_dir, feature_file), header=None)
    print(f"{feature_file} - has nan: {df.isnull().any().any()}")
    df.fillna(0, inplace=True)
    print(f"{feature_file} - has nan: {df.isnull().any().any()}")
    # Extract features excluding the first column (sample names)
    feature_array = df.iloc[:, 1:].values.astype(float)  # Shape: (num_samples, 4096)
    # Reshape each sample's 4096 features into a 64x64 matrix
    feature_array = feature_array.reshape(-1, 64, 64)  # Shape: (num_samples, 64, 64)
    new_features.append(feature_array)

# Stack the four channels to form a tensor of shape (num_samples, 4, 64, 64)
X_new = np.stack(new_features, axis=1)  # Shape: (num_samples, 4, 64, 64)

# Extract sample names (assuming the first column is sample names)
sample_names = []
for feature_file in new_feature_files:
    df = pd.read_csv(os.path.join(new_data_dir, feature_file), header=None)
    sample_names = df.iloc[:, 0].values  # Assuming all files have the same sample names and order
    break  # Only need to read once

# Convert to PyTorch tensor
X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

# Create DataLoader for new data
new_dataset = TensorDataset(X_new_tensor)
new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)

# =============================
# Step 5: Make Predictions on New Data
# =============================

predicted_scores = []
predicted_labels = []

with torch.no_grad():
    for (X_batch,) in new_loader:
        outputs = model(X_batch)
        predicted_scores.extend(outputs.cpu().numpy().flatten())

# Convert scores to binary labels (threshold = 0.5)
predicted_labels = [1 if score >= 0.5 else 0 for score in predicted_scores]

# =============================
# Step 6: Save Predictions
# =============================

# Create a DataFrame with sample names, predicted scores, and predicted labels
predictions_df = pd.DataFrame({
    'Sample': sample_names,
    'Predicted_Score': predicted_scores,
    'Predicted_Label': predicted_labels
})

# Save to CSV
predictions_df.to_csv(new_output_file, index=False)
print(f"Predictions saved to '{new_output_file}'")

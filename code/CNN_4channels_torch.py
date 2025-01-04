import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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


# Paths to the new feature CSV files and label file
feature_dir = '../data/DNA_features'
# feature_dir = '../data/data2_contigs'
# feature_dir = '../data/data2_contigs/kmer2'

feature_files = ['d_0_fraq_1865combined.csv', 'd_1_fraq_1865combined.csv', 'd_2_fraq_1865combined.csv', 'd_3_fraq_1865combined.csv']
# feature_files = ['contigs_3000_d0_combined.csv', 'contigs_3000_d1_combined.csv', 'contigs_3000_d2_combined.csv', 'contigs_3000_d3_combined.csv']

label_file = '../data/data1865_label.csv'
# label_file = '../data/contigs_label.csv'


predictions_dir = '../data/DNA_features/predictions'
# predictions_dir = '../data/data2_contigs/predictions/contig500'
# predictions_dir = '../data/data2_contigs/kmer2/predictions/contig4000'

# Read the label file
# Assuming the first column is the sample name and the second column is the label
label_df = pd.read_csv(label_file, header=None, names=['File', 'Label'])

# Initialize a list to hold feature arrays for each channel
features = []

for feature_file in feature_files:
    df = pd.read_csv(os.path.join(feature_dir, feature_file), header=None)
    # df.fillna(0, inplace=True)
    print(f"{feature_file} - has nan: {df.isnull().any().any()}")

    feature_array = df.iloc[:, 1:].values  # Shape: (1865, 4096)

    feature_array = feature_array.reshape(-1, 64, 64)  # Shape: (1865, 64, 64)
    # feature_array = feature_array.reshape(-1, 16, 16)  # Shape: (1865, 16, 16)

    features.append(feature_array)

X = np.stack(features, axis=1)  # Shape: (1865, 4, 64, 64)
y = label_df['Label'].values  # Shape: (1865,)


# Convert feature data to float32 tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

auc_scores = []
aupr_scores = []
acc_scores = []
recall_scores = []
precision_scores = []
specificity_scores = []
f1_scores = []
mcc_scores = []


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1345)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.fc1 = nn.Linear(32 * 15 * 15, 64)
        self.fc2 = nn.Linear(64, 1)  # Output single value for binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (batch, 16, 31, 31)
        x = self.pool(x)  # (batch, 16, 15, 15)
        x = torch.relu(self.conv2(x))  # (batch, 32, 12, 12)
        x = self.pool(x)  # (batch, 32, 6, 6)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 32*6*6)
        x = torch.relu(self.fc1(x))  # (batch, 64)
        x = torch.sigmoid(self.fc2(x))  # (batch, 1)
        return x

# kmer2
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)  # (batch, 16, 16, 16)
#         self.pool = nn.MaxPool2d(2, 2)  # (batch, 16, 8, 8)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (batch, 32, 8, 8)
#         self.pool2 = nn.MaxPool2d(2, 2)  # (batch, 32, 4, 4)
#
#         # Compute the flattened size for the fully connected layer:
#         self.fc1 = nn.Linear(32 * 4 * 4, 64)
#         self.fc2 = nn.Linear(64, 1)  # Output single value for binary classification
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))  # (batch, 16, 16, 16)
#         x = self.pool(x)  # (batch, 16, 8, 8)
#         x = torch.relu(self.conv2(x))  # (batch, 32, 8, 8)
#         x = self.pool2(x)  # (batch, 32, 4, 4)
#         x = x.view(x.size(0), -1)  # Flatten: (batch, 32*4*4)
#         x = torch.relu(self.fc1(x))  # (batch, 64)
#         x = torch.sigmoid(self.fc2(x))  # (batch, 1)
#         return x


for fold, (train_index, test_index) in enumerate(kfold.split(X, y), 1):
    X_train, X_test = X_tensor[train_index], X_tensor[test_index]
    y_train, y_test = y_tensor[train_index], y_tensor[test_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = ConvNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    for epoch in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())


    def evaluate_metrics(y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        auc = roc_auc_score(y_true, y_pred)
        aupr = average_precision_score(y_true, y_pred)
        y_pred_binary = np.round(y_pred)
        acc = accuracy_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_test, np.round(y_pred))
        return auc, aupr, acc, recall, precision, specificity, f1, mcc


    auc, aupr, acc, recall, precision, specificity, f1, mcc= evaluate_metrics(y_true, y_pred)

    auc_scores.append(auc)
    aupr_scores.append(aupr)
    acc_scores.append(acc)
    recall_scores.append(recall)
    precision_scores.append(precision)
    specificity_scores.append(specificity)
    f1_scores.append(f1)
    mcc_scores.append(mcc)

    predictions_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    predictions_df.to_csv(os.path.join(predictions_dir, f'fold_{fold}_predictions.csv'), index=False)
    print(
        f"Fold {fold} - AUC: {auc:.4f}, AUPR: {aupr:.4f}, Accuracy: {acc:.4f}, "
        f"Recall: {recall:.4f}, Precision: {precision:.4f}, "
        f"Specificity: {specificity:.4f}, F1 Score: {f1:.4f}, mcc Score: {mcc:.4f}"
    )

print("\nAverage Metrics Across All Folds:")
print(f"Average AUC: {np.mean(auc_scores):.4f}")
print(f"Average AUPR: {np.mean(aupr_scores):.4f}")
print(f"Average Accuracy: {np.mean(acc_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Specificity: {np.mean(specificity_scores):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
print(f"Average mcc: {np.mean(mcc_scores):.4f}")

metrics_df = pd.DataFrame({
    'Fold': range(1, 6),
    'AUC': auc_scores,
    'AUPR': aupr_scores,
    'Accuracy': acc_scores,
    'Recall': recall_scores,
    'Precision': precision_scores,
    'Specificity': specificity_scores,
    'F1 Score': f1_scores,
    'MCC': mcc_scores
})

metrics_df.to_csv(os.path.join(predictions_dir, 'evaluation_metrics.csv'), index=False)


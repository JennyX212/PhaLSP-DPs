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
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

feature_dir = '../data/DNA_features'
feature_files = ['d_0_fraq_1865combined.csv', 'd_1_fraq_1865combined.csv', 'd_2_fraq_1865combined.csv', 'd_3_fraq_1865combined.csv']
label_file = '../data/data1865_label.csv'
qs_feature_file = '../data/QS_data/1865phage_qs_feature.csv'  # QS特征文件路径
# text_feature_file = '../data/Bert/1865_transformer_representation.csv'  # 文本特征文件路径
predictions_dir = '../data/predictions'

label_df = pd.read_csv(label_file, header=None, names=['File', 'Label'])

qs_df = pd.read_csv(qs_feature_file)
qs_features = qs_df.iloc[:, 1:].values  # Shape: (1865, 35)

features = []

for feature_file in feature_files:
    df = pd.read_csv(os.path.join(feature_dir, feature_file), header=None)
    print(f"{feature_file} - has nan: {df.isnull().any().any()}")
    df.fillna(0, inplace=True)
    print(f"{feature_file} - has nan: {df.isnull().any().any()}")

    feature_array = df.iloc[:, 1:].values  # Shape: (1865, 4096)
    feature_array = feature_array.reshape(-1, 64, 64)  # Shape: (1865, 64, 64)
    features.append(feature_array)

X = np.stack(features, axis=1)  # Shape: (1865, 4, 64, 64)
y = label_df['Label'].values  # Shape: (1865,)

# Step 2: Convert to PyTorch Tensors

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Convert QS features to float32 tensors
qs_tensor = torch.tensor(qs_features, dtype=torch.float32)

# Initialize lists to store evaluation metrics
auc_scores = []
aupr_scores = []
acc_scores = []
recall_scores = []
precision_scores = []
specificity_scores = []
f1_scores = []
mcc_scores = []

# Initialize lists to store ROC and PR curves
roc_curves = []  # List to store interpolated TPR for each fold
pr_curves = []   # List to store interpolated Precision for each fold

# Define 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Define the CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, padding=1)  # Input channels: 3,  nn.Conv2d(3, 16, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.fc1 = nn.Linear(32 * 15 * 15 + 35, 64)  # +35 for QS features
        self.fc2 = nn.Linear(64, 1)  # Output single value for binary classification

    def forward(self, x, qs):
        x = torch.relu(self.conv1(x))  # (batch, 16, 31, 31)
        x = self.pool(x)               # (batch, 16, 15, 15)
        x = torch.relu(self.conv2(x))  # (batch, 32, 12, 12)
        x = self.pool(x)               # (batch, 32, 6, 6)
        x = x.view(x.size(0), -1)      # Flatten: (batch, 32*6*6)
        x = torch.relu(self.fc1(torch.cat((x, qs), dim=1)))  # Concatenate and apply fc1
        x = torch.sigmoid(self.fc2(x)) # (batch, 1)
        return x

# Define fixed FPR and Recall points for averaging ROC and PR curves
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kfold.split(X, y), 1):
    print(f"\nProcessing Fold {fold}...")
    # Split the data into training and testing sets for this fold
    X_train, X_test = X_tensor[train_index], X_tensor[test_index]
    y_train, y_test = y_tensor[train_index], y_tensor[test_index]
    qs_train, qs_test = qs_tensor[train_index], qs_tensor[test_index]  # QS特征

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, qs_train, y_train)
    test_dataset = TensorDataset(X_test, qs_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = ConvNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(100):
        model.train()
        for X_batch, qs_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            qs_batch = qs_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, qs_batch)  # 传递QS特征
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, qs_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            qs_batch = qs_batch.to(device)
            outputs = model(X_batch, qs_batch)  # 传递QS特征
            y_true.extend(y_batch.cpu().numpy().flatten().astype(int))
            y_pred.extend(outputs.cpu().numpy().flatten().astype(float))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# label_df = pd.read_csv(label_file, header=None, names=['File', 'Label'])
# qs_df = pd.read_csv(qs_feature_file)
# qs_features = qs_df.iloc[:, 1:].values  # Shape: (1865, 35)
#
# # Read Text features
# text_df = pd.read_csv(text_feature_file)
# text_features = text_df.iloc[:, 1:].values  # 假设文本特征从第二列开始，形状为 (1865, text_feature_dim)
#
# features = []
# for feature_file in feature_files:
#     # Read each feature CSV
#     df = pd.read_csv(os.path.join(feature_dir, feature_file), header=None)
#     print(f"{feature_file} - has nan: {df.isnull().any().any()}")
#     df.fillna(0, inplace=True)
#     print(f"{feature_file} - has nan: {df.isnull().any().any()}")
#
#     feature_array = df.iloc[:, 1:].values  # Shape: (1865, 4096)
#     feature_array = feature_array.reshape(-1, 64, 64)  # Shape: (1865, 64, 64)
#     features.append(feature_array)
#
# # Stack the four channels to form a tensor of shape (1865, 4, 64, 64)
# X_image = np.stack(features, axis=1)  # Shape: (1865, 4, 64, 64)
#
# # Extract labels
# y = label_df['Label'].values  # Shape: (1865,)
#
# # Step : Convert to PyTorch Tensors
# X_image_tensor = torch.tensor(X_image, dtype=torch.float32)
#
# # Convert QS features to float32 tensors
# qs_tensor = torch.tensor(qs_features, dtype=torch.float32)
# # Convert Text features to float32 tensors
# text_tensor = torch.tensor(text_features, dtype=torch.float32)
# # Convert labels to float32 tensors and reshape to (1865, 1)
# y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
#
# # Initialize lists to store evaluation metrics
# auc_scores = []
# aupr_scores = []
# acc_scores = []
# recall_scores = []
# precision_scores = []
# specificity_scores = []
# f1_scores = []
# mcc_scores = []
#
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
#
# # Define the CNN model
# class ConvNet(nn.Module):
#     def __init__(self, qs_feature_dim, text_feature_dim):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(4, 16, kernel_size=4, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
#         self.fc1 = nn.Linear(32 * 15 * 15 + qs_feature_dim + text_feature_dim, 64)
#         self.fc2 = nn.Linear(64, 1)  # Output single value for binary classification
#
#     def forward(self, x, qs, text):
#         x = torch.relu(self.conv1(x))  # (batch, 16, 31, 31)
#         x = self.pool(x)               # (batch, 16, 15, 15)
#         x = torch.relu(self.conv2(x))  # (batch, 32, 12, 12)
#         x = self.pool(x)               # (batch, 32, 6, 6)
#         x = x.view(x.size(0), -1)      # Flatten: (batch, 32*6*6)
#         x = torch.cat((x, qs, text), dim=1)  # 拼接图像特征、QS特征和文本特征
#         x = torch.relu(self.fc1(x))    # (batch, 64)
#         x = torch.sigmoid(self.fc2(x)) # (batch, 1)
#         return x
#
# qs_feature_dim = qs_features.shape[1]  #
# text_feature_dim = text_features.shape[1]  # N
#
# for fold, (train_index, test_index) in enumerate(kfold.split(X_image, y), 1):
#     print(f"\nProcessing Fold {fold}...")
#     X_train_image, X_test_image = X_image_tensor[train_index], X_image_tensor[test_index]
#     qs_train, qs_test = qs_tensor[train_index], qs_tensor[test_index]  # QS特征
#     text_train, text_test = text_tensor[train_index], text_tensor[test_index]  # 文本特征
#     y_train, y_test = y_tensor[train_index], y_tensor[test_index]
#
#     train_dataset = TensorDataset(X_train_image, qs_train, text_train, y_train)
#     test_dataset = TensorDataset(X_test_image, qs_test, text_test, y_test)
#
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#     # Initialize the model, loss function, and optimizer
#     model = ConvNet(qs_feature_dim=qs_feature_dim, text_feature_dim=text_feature_dim)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     # Training loop
#     for epoch in range(100):
#         model.train()
#         for X_batch, qs_batch, text_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             qs_batch = qs_batch.to(device)
#             text_batch = text_batch.to(device)
#             y_batch = y_batch.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(X_batch, qs_batch, text_batch)  # 传递QS特征和文本特征
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#
#     # Evaluation
#     model.eval()
#     y_true = []
#     y_pred = []
#
#     with torch.no_grad():
#         for X_batch, qs_batch, text_batch, y_batch in test_loader:
#             X_batch = X_batch.to(device)
#             qs_batch = qs_batch.to(device)
#             text_batch = text_batch.to(device)
#             outputs = model(X_batch, qs_batch, text_batch)  # 传递QS特征和文本特征
#             y_true.extend(y_batch.cpu().numpy().flatten().astype(int))
#             y_pred.extend(outputs.cpu().numpy().flatten().astype(float))
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def evaluate_metrics(y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        auc = roc_auc_score(y_true, y_pred)
        aupr = average_precision_score(y_true, y_pred)
        y_pred_binary = np.round(y_pred)
        acc = accuracy_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0
        f1 = f1_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
        return auc, aupr, acc, recall, precision, specificity, f1, mcc, fpr, tpr, precision_vals, recall_vals

    auc, aupr, acc, recall, precision, specificity, f1, mcc, fpr, tpr, precision_vals, recall_vals = evaluate_metrics(y_true, y_pred)

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
        f"Specificity: {specificity:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}"
    )


#  Save results

print("\nAverage Metrics Across All Folds:")
print(f"Average AUC: {np.mean(auc_scores):.4f}")
print(f"Average AUPR: {np.mean(aupr_scores):.4f}")
print(f"Average Accuracy: {np.mean(acc_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Specificity: {np.mean(specificity_scores):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
print(f"Average MCC: {np.mean(mcc_scores):.4f}")

# Save evaluation metrics
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


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import os

csv_file_path = '../data/Bert/seq500_label.csv'  # 修改实际路径
try:
    data = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"文件未找到，请检查路径是否正确：{csv_file_path}")
    exit(1)

class PhageDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # Shape: (max_len)
        attention_mask = encoding['attention_mask'].squeeze()  # Shape: (max_len)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.float32)

sequences = data['sequence'].tolist()
labels = data['label'].tolist()

unique_labels = set(labels)
if unique_labels != {0, 1}:
    print("警告：标签不是二分类的。请确保标签为0和1。")

tokenizer_path = '../bert-base-uncased'  # 确保此路径包含必要的文件，如vocab.txt等
try:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    print(f"加载tokenizer失败：{e}")
    exit(1)

max_seq_len = 500  # BERT的最大序列长度是512，这里设置为500
print(f"使用的最大序列长度: {max_seq_len}")

dataset = PhageDataset(sequences, labels, tokenizer, max_len=max_seq_len)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class PhageTransformer(nn.Module):
    def __init__(self, bert_model_path):
        super(PhageTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)  # 输出1个值用于二分类

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 使用[CLS] token的表示作为句子的表示
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

model = PhageTransformer(bert_model_path=tokenizer_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
epochs = 50
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss()

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    true_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    preds = np.array(preds) >= 0.5
    true_labels = np.array(true_labels)
    accuracy = accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, torch.sigmoid(torch.tensor(preds)).numpy())
    f1 = f1_score(true_labels, preds)
    return avg_loss, accuracy, auc, f1

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
    print(f"训练损失: {train_loss:.4f}")

    val_loss, val_accuracy, val_auc, val_f1 = eval_model(model, val_loader, criterion, device)
    print(f"验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, AUC: {val_auc:.4f}, F1分数: {val_f1:.4f}\n")

model_save_path = '../data/Bert/phage_transformer_trained.pth'
torch.save(model.state_dict(), model_save_path)
print(f"训练好的模型已保存到 {model_save_path}")

class FeatureExtractor(nn.Module):
    def __init__(self, bert_model_path):
        super(FeatureExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        return pooled_output  # 返回的是pooled_output作为特征表示


feature_extractor = FeatureExtractor(bert_model_path=tokenizer_path)
feature_extractor.load_state_dict(model.bert.state_dict())  # 加载BERT部分的权重
feature_extractor.to(device)
feature_extractor.eval()

# 获取所有样本的特征表示
representations = []
samples = data['sample'].tolist()

with torch.no_grad():
    for input_ids, attention_mask, _ in DataLoader(dataset, batch_size=32, shuffle=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        features = feature_extractor(input_ids, attention_mask)
        features = features.cpu().numpy()
        representations.extend(features)

output_df = pd.DataFrame(representations)
output_df.insert(0, 'sample', samples)  # 将'sample'列放在第一列

# 保存表征到CSV
output_csv_path = '../data/Bert/1865_transformer_representation_trained.csv'
output_df.to_csv(output_csv_path, index=False)
print(f"特征表示已保存到 {output_csv_path}")





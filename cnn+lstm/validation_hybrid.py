import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')

# global config
MODEL_TYPE = 'HYBRID'  

# Local paths logic
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'cnn', 'landmarks.npz')
META_PATH = os.path.join(BASE_DIR, 'cnn', 'landmarks_metadata.json')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'validation_results_{MODEL_TYPE.lower()}')

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'k_folds': 5,
    'batch_size': 32,
    'epochs_per_fold': 50,
    'fine_tune_epochs': 80,
    'learning_rate': 0.001 if MODEL_TYPE == '1DCNN' else 0.0005,
    'seed': 42
}

#model architectures

class SignLanguageCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64); self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2); self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128); self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2); self.drop2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256); self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.global_pool(self.relu3(self.bn3(self.conv3(x))))
        return self.classifier(x)

class SignLanguageHybrid(nn.Module):
    def __init__(self, num_classes, input_dim=126, hidden_dim=256, num_lstm_layers=2):
        super(SignLanguageHybrid, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU()
        )
        self.lstm = nn.LSTM(128, hidden_dim, num_lstm_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_target = torch.full_like(pred, self.smoothing / (self.classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-smooth_target * torch.log_softmax(pred, dim=1), dim=1))

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_data():
    data = np.load(DATA_PATH)
    X, y = data['X'].astype(np.float32), data['y'].astype(np.int64)
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)
    if len(X.shape) == 4: X = X.reshape(X.shape[0], X.shape[1], -1)
    if MODEL_TYPE == '1DCNN': X = np.transpose(X, (0, 2, 1))
    return X, y, metadata


# training and evaluation functions
def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        if MODEL_TYPE == 'HYBRID': torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += batch_y.size(0)
        correct += pred.eq(batch_y).sum().item()
    if scheduler: scheduler.step()
    return running_loss/len(loader), 100.*correct/total

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# validation and fine tuning
def main():
    X, y, metadata = load_data()
    num_classes = len(metadata['word_to_idx'])
    idx_to_word = {int(k): v for k, v in metadata['idx_to_word'].items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['seed'])
    overall_preds, overall_labels, fold_accs = [], [], []

    print(f"\nstarting {CONFIG['k_folds']}-fold cv for {MODEL_TYPE}...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_loader = DataLoader(GestureDataset(X[train_idx], y[train_idx]), batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(GestureDataset(X[val_idx], y[val_idx]), batch_size=CONFIG['batch_size'], shuffle=False)

        if MODEL_TYPE == '1DCNN':
            model = SignLanguageCNN1D(X.shape[1], num_classes).to(device)
        else:
            model = SignLanguageHybrid(num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

        best_fold_acc = 0
        for epoch in range(CONFIG['epochs_per_fold']):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            preds, labels = evaluate(model, val_loader, device)
            val_acc = accuracy_score(labels, preds) * 100
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'temp_fold{fold+1}.pth'))

        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'temp_fold{fold+1}.pth')))
        f_preds, f_labels = evaluate(model, val_loader, device)
        overall_preds.extend(f_preds); overall_labels.extend(f_labels); fold_accs.append(best_fold_acc)
        print(f"Fold {fold+1} Best Acc: {best_fold_acc:.2f}%")

    # generate Reports
    generate_reports(overall_labels, overall_preds, idx_to_word, fold_accs)

    # fine tuning
    print(f"\nfine-tuning with optimized strategy...")
    dataset = GestureDataset(X, y)
    train_size = int(0.85 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    if MODEL_TYPE == '1DCNN':
        model = SignLanguageCNN1D(X.shape[1], num_classes).to(device)
    else:
        model = SignLanguageHybrid(num_classes).to(device)

    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)

    if MODEL_TYPE == 'HYBRID':
        params = [
            {'params': model.feature_extractor.parameters(), 'lr': CONFIG['learning_rate'] * 0.5},
            {'params': model.lstm.parameters(), 'lr': CONFIG['learning_rate'] * 0.5},
            {'params': model.classifier.parameters(), 'lr': CONFIG['learning_rate']}
        ]
    else:
        params = model.parameters()

    optimizer = optim.AdamW(params, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_ft_acc = 0
    for epoch in range(CONFIG['fine_tune_epochs']):
        train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        preds, labels = evaluate(model, val_loader, device)
        ft_acc = accuracy_score(labels, preds) * 100
        if ft_acc > best_ft_acc:
            best_ft_acc = ft_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_finetuned.pth'))
        if (epoch+1) % 20 == 0: print(f"Fine-tune Epoch {epoch+1:02d} | Val Acc: {ft_acc:.1f}%")

    print(f"\nFinal Results: K-Fold Mean: {np.mean(fold_accs):.2f}% | Fine-tuned: {best_ft_acc:.2f}%")

def generate_reports(y_true, y_pred, idx_to_word, fold_accs):
    target_names = [idx_to_word[i] for i in sorted(idx_to_word.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png')); plt.close()

if __name__ == "__main__":
    main()
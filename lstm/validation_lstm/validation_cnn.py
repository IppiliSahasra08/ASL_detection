""""""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')
MODEL_TYPE = '1DCNN'
DRIVE_PATH = '/content/drive/MyDrive'
DATA_PATH = os.path.join(DRIVE_PATH, 'landmarks.npz')
META_PATH = os.path.join(DRIVE_PATH, 'landmarks_metadata.json')
OUTPUT_DIR = os.path.join(DRIVE_PATH, f'validation_results_{MODEL_TYPE.lower()}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
drive.mount('/content/drive')
CONFIG = {'k_folds': 5, 'batch_size': 32, 'epochs_per_fold': 50, 'learning_rate': 0.001 if MODEL_TYPE == '1DCNN' else 0.0005, 'seed': 42}
class SignLanguageCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.global_pool(self.relu3(self.bn3(self.conv3(x))))
        return self.classifier(x)
class SignLanguageHybrid(nn.Module):
    def __init__(self, num_classes, input_dim=126, hidden_dim=256, num_lstm_layers=2):
        super(SignLanguageHybrid, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU())
        self.lstm = nn.LSTM(128, hidden_dim, num_lstm_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
def load_and_preprocess():
    print(f'Loading data for {MODEL_TYPE}...')
    data = np.load(DATA_PATH)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)
    if len(X.shape) == 4:
        X = X.reshape(X.shape[0], X.shape[1], -1)
    if MODEL_TYPE == '1DCNN':
        X = np.transpose(X, (0, 2, 1))
    return (X, y, metadata)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = (0, 0, 0)
    for batch_x, batch_y in loader:
        batch_x, batch_y = (batch_x.to(device), batch_y.to(device))
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        if MODEL_TYPE == 'HYBRID':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += batch_y.size(0)
        correct += pred.eq(batch_y).sum().item()
    return (running_loss / len(loader), 100.0 * correct / total)
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = ([], [])
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = (batch_x.to(device), batch_y.to(device))
            outputs = model(batch_x)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return (np.array(all_preds), np.array(all_labels))
def run_validation():
    X, y, metadata = load_and_preprocess()
    num_classes = len(metadata['word_to_idx'])
    idx_to_word = {int(k): v for k, v in metadata['idx_to_word'].items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['seed'])
    overall_preds, overall_labels = ([], [])
    fold_accuracies = []
    print(f"\nStarting {CONFIG['k_folds']}-Fold Validation for {MODEL_TYPE}...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\n--- FOLD {fold + 1} ---')
        train_sub = SubsetRandomSampler(train_idx)
        val_sub = SubsetRandomSampler(val_idx)
        full_ds = GestureDataset(X, y)
        train_loader = DataLoader(full_ds, batch_size=CONFIG['batch_size'], sampler=train_sub)
        val_loader = DataLoader(full_ds, batch_size=CONFIG['batch_size'], sampler=val_sub)
        if MODEL_TYPE == '1DCNN':
            model = SignLanguageCNN1D(in_channels=X.shape[1], num_classes=num_classes).to(device)
        else:
            model = SignLanguageHybrid(num_classes=num_classes).to(device)
        cw = compute_class_weight('balanced', classes=np.unique(y[train_idx]), y=y[train_idx])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
        best_fold_acc = 0
        for epoch in range(CONFIG['epochs_per_fold']):
            loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            preds, labels = evaluate(model, val_loader, device)
            val_acc = accuracy_score(labels, preds) * 100
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_model_fold{fold + 1}.pth'))
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1:02d} | Loss: {loss:.4f} | Train Acc: {acc:.1f}% | Val Acc: {val_acc:.1f}%')
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'best_model_fold{fold + 1}.pth')))
        f_preds, f_labels = evaluate(model, val_loader, device)
        overall_preds.extend(f_preds)
        overall_labels.extend(f_labels)
        fold_accuracies.append(best_fold_acc)
        print(f'Fold {fold + 1} Finished. Best Acc: {best_fold_acc:.2f}%')
    generate_reports(overall_labels, overall_preds, idx_to_word, fold_accuracies)
def generate_reports(y_true, y_pred, idx_to_word, fold_accs):
    target_names = [idx_to_word[i] for i in sorted(idx_to_word.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print('\n' + '=' * 30 + '\nFINAL VALIDATION REPORT\n' + '=' * 30)
    print(f'Mean Accuracy: {np.mean(fold_accs):.2f}% (+/- {np.std(fold_accs):.2f}%)')
    print('\n', report)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f'MODEL TYPE: {MODEL_TYPE}\n')
        f.write(f'Mean Accuracy: {np.mean(fold_accs):.2f}%\n\n')
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix: {MODEL_TYPE}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.show()
if __name__ == '__main__':
    run_validation()

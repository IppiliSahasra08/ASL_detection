import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import time
import shutil
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# update these paths to your actual file locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'cnn', 'landmarks.npz')
META_PATH = os.path.join(BASE_DIR, 'cnn', 'landmarks_metadata.json')
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(SAVE_DIR, 'backups')

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# loading data and reshaping
print("Loading and preparing data...")
data = np.load(DATA_PATH)
X = data['X'].astype(np.float32)   
y = data['y'].astype(np.int64)

with open(META_PATH) as f:
    metadata = json.load(f)
NUM_CLASSES = len(metadata['word_to_idx'])

# reshaping for sequential processing
if len(X.shape) == 4:
    N, T, L, C = X.shape
    X = X.reshape(N, T, L * C)
    print(f"Reshaped X to: {X.shape} (N, Time, Features)")

# train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15/0.85, stratify=y_train, random_state=42)

# architechture
class SignLanguageHybrid(nn.Module):
    def __init__(self, num_classes, input_dim=126, hidden_dim=256, num_lstm_layers=2):
        super(SignLanguageHybrid, self).__init__()

        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )

        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, (hn, cn) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)

# dataset and training utils
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(GestureDataset(X_val, y_val), batch_size=32, shuffle=False)

# class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

model = SignLanguageHybrid(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# training loop
NUM_EPOCHS = 100
PATIENCE = 15
best_val_acc = 0.0
patience_counter = 0

print("\nStarting Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
        optimizer.step()

        train_loss += loss.item()
        _, pred = outputs.max(1)
        train_total += batch_y.size(0)
        train_correct += pred.eq(batch_y).sum().item()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, pred = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += pred.eq(batch_y).sum().item()

    val_acc = 100 * val_correct / val_total
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {100*train_correct/train_total:.1f}% | Val Acc: {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        save_path = os.path.join(SAVE_DIR, 'best_model_hybrid.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'val_acc': val_acc
        }, save_path)
        shutil.copy(save_path, os.path.join(BACKUP_DIR, 'best_model_hybrid.pth'))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}%")
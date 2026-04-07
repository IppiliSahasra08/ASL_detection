import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import time
import os
# import shutil (removed as not needed for local)

# 1d cnn training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'landmarks.npz')
META_PATH = os.path.join(SCRIPT_DIR, 'landmarks_metadata.json')
SAVE_DIR = SCRIPT_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

# load and prep data
print("\nLoading data...")
data = np.load(DATA_PATH)
X = data['X'].astype(np.float32)  
y = data['y'].astype(np.int64)

with open(META_PATH, 'r') as f:
    metadata = json.load(f)

NUM_CLASSES = len(metadata['word_to_idx'])
print(f"  Initial X shape : {X.shape}")
print(f"  Classes         : {NUM_CLASSES}")

X = np.transpose(X, (0, 2, 1))
print(f"  Reshaped X      : {X.shape} -> (Batch, Features, Time)")

IN_FEATURES = X.shape[1] 
SEQ_LENGTH = X.shape[2]  

class SignLanguage1DDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# training split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n  Train samples : {len(X_train)}")
print(f"  Val samples   : {len(X_val)}")

train_loader = DataLoader(SignLanguage1DDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(SignLanguage1DDataset(X_val, y_val), batch_size=32, shuffle=False)

# class weigts
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# model architecture
class SignLanguageCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(0.3) 

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.global_pool(self.relu3(self.bn3(self.conv3(x))))
        x = self.classifier(x)
        return x

model = SignLanguageCNN1D(in_channels=IN_FEATURES, num_classes=NUM_CLASSES).to(device)
print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# training loop
NUM_EPOCHS = 75
PATIENCE = 15 
patience_counter = 0
best_val_acc = 0.0

# training

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += batch_y.size(0)
        train_correct += predicted.eq(batch_y).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total

    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += predicted.eq(batch_y).sum().item()

    val_acc = 100 * val_correct / val_total
    scheduler.step(val_acc)

    epoch_time = time.time() - epoch_start

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        marker = "BEST"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'num_classes': NUM_CLASSES,
            'in_features': IN_FEATURES,
            'metadata': metadata
        }, os.path.join(SAVE_DIR, 'best_model_1dcnn.pth'))

    else:
        patience_counter += 1
        marker = ""

    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} ({epoch_time:.1f}s) | "
          f"Train: {train_loss:.4f}/{train_acc:5.1f}% | "
          f"Val: {val_acc:5.1f}%{marker}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print("\ntraining complete")
print(f"Model saved to: {os.path.join(SAVE_DIR, 'best_model_1dcnn.pth')}")
print("=" * 60)
"""
Sign Language Recognition - Comprehensive Validation and Classification Report
==============================================================================
This script performs:
1. K-Fold Cross-Validation (5-fold stratified)
2. Detailed per-class classification report
3. Confusion matrix visualization
4. Fine-tuning with optimized settings
All outputs are saved to the Google Drive validation_lstm folder.
Author: MiniMax Agent
Project: Sign Language Translator using Bi-LSTM with Attention
FIXED: Updated to use word_to_idx and idx_to_word (matching your metadata format)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
import json
import os
from google.colab import drive
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
print("\n" + "="*70)
print("  SETTING UP GOOGLE DRIVE")
print("="*70)
drive.mount('/content/drive')
print("Google Drive mounted successfully!")
WORK_DIR = '/content/drive/My Drive/validation_lstm'
os.chdir(WORK_DIR)
print(f"Working directory: {WORK_DIR}")
print("\nCurrent folder contents:")
for item in os.listdir(WORK_DIR):
    print(f"  - {item}")
print("\n" + "="*70)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
CONFIG = {
    'k_folds': 5,
    'num_frames': 30,
    'feature_dim': 126,  
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs_per_fold': 50,
    'early_stopping_patience': 10,
    'num_classes': 10  
}
class Attention(nn.Module):
    """Attention mechanism for sequence weighting - helps model focus on important frames"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output, attention_weights
class SignLanguageLSTM(nn.Module):
    """Bi-LSTM with Attention mechanism for sign language recognition"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignLanguageLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = Attention(hidden_size * 2)  
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        self._init_weights()
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attention_weights
def load_processed_data(data_path='landmarks.npz', metadata_path='landmarks_metadata.json'):
    """Load preprocessed landmark data and metadata from current directory
    FIXED: Now uses word_to_idx and idx_to_word instead of label_to_idx and idx_to_label
    """
    print("\n" + "="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)
    data = np.load(data_path)
    X = data['X']  
    y = data['y']  
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    word_to_idx = metadata['word_to_idx']
    idx_to_word = {int(k): v for k, v in metadata['idx_to_word'].items()}
    print(f"Loaded {len(X)} samples")
    print(f"Shape: {X.shape} (samples × frames × features)")
    print(f"Number of classes: {len(word_to_idx)}")
    print(f"Classes: {list(word_to_idx.keys())}")
    return X, y, word_to_idx, idx_to_word
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs, _ = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy
def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
def run_k_fold_validation(X, y, word_to_idx, idx_to_word, config):
    """
    Perform K-Fold Cross-Validation
    Uses Stratified K-Fold to maintain class distribution in each fold.
    This ensures each fold has the same proportion of each class.
    All fold models are saved to Google Drive.
    """
    print("\n" + "="*60)
    print("K-FOLD CROSS VALIDATION")
    print("="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of folds: {config['k_folds']}")
    print(f"Total samples: {len(X)}\n")
    skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=SEED)
    fold_results = []
    all_val_preds = []
    all_val_labels = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{config['k_folds']}")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
        train_dist = defaultdict(int)
        val_dist = defaultdict(int)
        for idx in train_idx:
            train_dist[idx_to_word[y[idx]]] += 1
        for idx in val_idx:
            val_dist[idx_to_word[y[idx]]] += 1
        print("Training class distribution: " + 
              ", ".join([f"{k}:{v}" for k, v in train_dist.items()]))
        print("Validation class distribution: " + 
              ", ".join([f"{k}:{v}" for k, v in val_dist.items()]))
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=config['batch_size'],
            sampler=SubsetRandomSampler(train_idx),
            num_workers=0
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=config['batch_size'],
            sampler=SubsetRandomSampler(val_idx),
            num_workers=0
        )
        model = SignLanguageLSTM(
            input_size=config['feature_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        best_val_acc = 0
        patience_counter = 0
        fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        for epoch in range(config['epochs_per_fold']):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = evaluate_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_acc'].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(WORK_DIR, f'best_model_fold{fold+1}.pth')
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
            if patience_counter >= config['early_stopping_patience']:
                print(f"\n  -> Early stopping at epoch {epoch+1}")
                break
        fold_results.append({
            'fold': fold + 1,
            'best_val_accuracy': best_val_acc,
            'history': fold_history
        })
        model_path = os.path.join(WORK_DIR, f'best_model_fold{fold+1}.pth')
        model.load_state_dict(torch.load(model_path))
        _, _, val_preds, val_labels = evaluate_epoch(model, val_loader, criterion, device)
        all_val_preds.extend(val_preds)
        all_val_labels.extend(val_labels)
        print(f"\n  -> Fold {fold+1} Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("\n" + "="*60)
    print("K-FOLD VALIDATION SUMMARY")
    print("="*60)
    accuracies = [r['best_val_accuracy'] for r in fold_results]
    print(f"\nIndividual Fold Accuracies:")
    for i, acc in enumerate(accuracies):
        print(f"  Fold {i+1}: {acc*100:.2f}%")
    print(f"\nMean Accuracy: {np.mean(accuracies)*100:.2f}% (+/- {np.std(accuracies)*100:.2f}%)")
    print(f"Min Accuracy:  {np.min(accuracies)*100:.2f}%")
    print(f"Max Accuracy:  {np.max(accuracies)*100:.2f}%")
    print(f"95% CI: [{np.mean(accuracies)*100 - 1.96*np.std(accuracies)*100:.2f}%, "
          f"{np.mean(accuracies)*100 + 1.96*np.std(accuracies)*100:.2f}%]")
    return fold_results, np.array(all_val_preds), np.array(all_val_labels)
def generate_per_class_report(y_true, y_pred, idx_to_word, save_path='classification_report.txt'):
    """
    Generate detailed per-class classification report
    This provides comprehensive metrics for each sign language word:
    - Precision: How many predicted as this class are correct
    - Recall: How many actual this class are correctly identified
    - F1-Score: Harmonic mean of precision and recall (balance both)
    - Support: Number of samples in each class
    Report is saved to Google Drive.
    """
    print("\n" + "="*60)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("="*60)
    target_names = [idx_to_word[i] for i in sorted(idx_to_word.keys())]
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        digits=4,
        output_dict=False
    )
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        digits=4,
        output_dict=True
    )
    print("\n" + report)
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SIGN LANGUAGE RECOGNITION - DETAILED CLASSIFICATION REPORT\n")
        f.write("Generated: " + str(os.path.getmtime(save_path)) + "\n")
        f.write("="*70 + "\n\n")
        f.write("METHODOLOGY:\n")
        f.write("-" * 70 + "\n")
        f.write("This report was generated using 5-Fold Stratified Cross-Validation.\n")
        f.write("Each sample was validated exactly once across all folds.\n")
        f.write("All results are saved to Google Drive.\n\n")
        f.write("METRIC DEFINITIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("• Precision: Proportion of positive predictions that were correct\n")
        f.write("• Recall: Proportion of actual positives that were correctly identified\n")
        f.write("• F1-Score: Harmonic mean of precision and recall (balance both)\n")
        f.write("• Support: Number of samples in each class\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 70 + "\n")
        f.write(report)
        f.write("\n\n")
        f.write("PER-CLASS ANALYSIS:\n")
        f.write("-" * 70 + "\n")
        for class_name in target_names:
            metrics = report_dict[class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics['precision']*100:.2f}%\n")
            f.write(f"  Recall:    {metrics['recall']*100:.2f}%\n")
            f.write(f"  F1-Score:  {metrics['f1-score']*100:.2f}%\n")
            f.write(f"  Support:   {int(metrics['support'])} samples\n")
        f.write("\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 70 + "\n")
        f.write("Rows: Actual | Columns: Predicted\n\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"{'True \\ Pred':<12}")
        for name in target_names:
            f.write(f" {name[:10]:>10}")
        f.write("\n")
        f.write("-" * (12 + 10 * len(target_names)) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{target_names[i][:10]:<12}")
            for val in row:
                f.write(f" {val:>10}")
            f.write("\n")
        f.write("\n\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write("• Diagonal values: Correct predictions\n")
        f.write("• Off-diagonal values: Misclassifications\n")
        f.write("• Higher diagonal values indicate better performance\n")
        f.write("• Check off-diagonal patterns to identify confused class pairs\n")
    print(f"\nReport saved to: {save_path}")
    return report_dict, report, cm
def plot_confusion_matrix(cm, idx_to_word, save_path='confusion_matrix.png'):
    """Visualize confusion matrix with annotations - saved to Google Drive"""
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRIX VISUALIZATION")
    print("="*60)
    plt.figure(figsize=(14, 12))
    target_names = [idx_to_word[i] for i in sorted(idx_to_word.keys())]
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={'label': 'Number of Predictions'},
        annot_kws={'size': 12}
    )
    plt.title('Confusion Matrix - Sign Language Recognition\n(5-Fold Cross-Validation)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
def plot_per_class_metrics(report_dict, idx_to_word, save_path='per_class_metrics.png'):
    """Visualize per-class precision, recall, and F1-score - saved to Google Drive"""
    print("\n" + "="*60)
    print("GENERATING PER-CLASS METRICS VISUALIZATION")
    print("="*60)
    target_names = [idx_to_word[i] for i in sorted(idx_to_word.keys())]
    metrics = {
        'Precision': [], 'Recall': [], 'F1-Score': []
    }
    for name in target_names:
        metrics['Precision'].append(report_dict[name]['precision'] * 100)
        metrics['Recall'].append(report_dict[name]['recall'] * 100)
        metrics['F1-Score'].append(report_dict[name]['f1-score'] * 100)
    x = np.arange(len(target_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width, metrics['Precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, metrics['Recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', color='#e74c3c')
    ax.set_xlabel('Sign Language Words', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Performance Metrics\n(5-Fold Cross-Validation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved to: {save_path}")
def fine_tune_model(X, y, word_to_idx, idx_to_word, config, num_epochs=80):
    """
    Fine-tune the model with optimized settings
    Key improvements:
    1. Label smoothing - reduces overconfidence
    2. Differential learning rates - lower for LSTM, higher for classifier
    3. Cosine annealing with warm restarts - better optimization
    4. Weight decay - prevents overfitting
    Fine-tuned model is saved to Google Drive.
    """
    print("\n" + "="*60)
    print("MODEL FINE-TUNING")
    print("="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx])),
        batch_size=config['batch_size'], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx])),
        batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    model_path = os.path.join(WORK_DIR, 'best_model.pth')
    if os.path.exists(model_path):
        model = SignLanguageLSTM(
            input_size=config['feature_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model from: {model_path}")
    else:
        model = SignLanguageLSTM(
            input_size=config['feature_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(device)
        print("No existing model found, training from scratch")
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
    criterion = LabelSmoothingLoss(config['num_classes'], smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': model.lstm.parameters(), 'lr': config['learning_rate'] * 0.1},
        {'params': model.attention.parameters(), 'lr': config['learning_rate'] * 0.5},
        {'params': model.classifier.parameters(), 'lr': config['learning_rate']}
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    best_val_acc = 0
    patience_counter = 0
    print(f"\nFine-tuning with {len(train_idx)} training and {len(val_idx)} validation samples")
    print("Strategy: Label smoothing (0.1), Differential LR, Cosine annealing\n")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        scheduler.step()
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | LR: {current_lr:.6f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            finetuned_path = os.path.join(WORK_DIR, 'best_model_finetuned.pth')
            torch.save(model.state_dict(), finetuned_path)
            print(f"  -> New best model! Accuracy: {val_acc:.1f}%")
        else:
            patience_counter += 1
        if patience_counter >= 25:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    print(f"\nFine-tuning complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {finetuned_path}")
    return best_val_acc
def main():
    print("\n" + "="*70)
    print("  SIGN LANGUAGE RECOGNITION - VALIDATION AND REPORTING SUITE  ")
    print("="*70)
    print("\nAll outputs will be saved to:")
    print(f"  {WORK_DIR}")
    print("\nThis script performs comprehensive model validation including:")
    print("  1. 5-Fold Cross-Validation")
    print("  2. Per-class Classification Report")
    print("  3. Confusion Matrix Visualization")
    print("  4. Per-class Metrics Visualization")
    print("  5. Model Fine-tuning")
    print("\n" + "-"*70)
    X, y, word_to_idx, idx_to_word = load_processed_data()
    fold_results, all_val_preds, all_val_labels = run_k_fold_validation(
        X, y, word_to_idx, idx_to_word, CONFIG
    )
    report_path = os.path.join(WORK_DIR, 'classification_report.txt')
    report_dict, report_text, cm = generate_per_class_report(
        all_val_labels, all_val_preds, idx_to_word, report_path
    )
    cm_path = os.path.join(WORK_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, idx_to_word, cm_path)
    metrics_path = os.path.join(WORK_DIR, 'per_class_metrics.png')
    plot_per_class_metrics(report_dict, idx_to_word, metrics_path)
    fine_tune_acc = fine_tune_model(X, y, word_to_idx, idx_to_word, CONFIG)
    summary = {
        'k_fold_results': [
            {
                'fold': r['fold'], 
                'accuracy': float(r['best_val_accuracy'])
            }
            for r in fold_results
        ],
        'mean_accuracy': float(np.mean([r['best_val_accuracy'] for r in fold_results])),
        'std_accuracy': float(np.std([r['best_val_accuracy'] for r in fold_results])),
        'fine_tune_accuracy': float(fine_tune_acc) / 100.0,
        'per_class_metrics': {
            label: {
                'precision': float(report_dict[label]['precision']),
                'recall': float(report_dict[label]['recall']),
                'f1_score': float(report_dict[label]['f1-score']),
                'support': int(report_dict[label]['support'])
            }
            for label in [idx_to_word[i] for i in sorted(idx_to_word.keys())]
        }
    }
    summary_path = os.path.join(WORK_DIR, 'validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*70)
    print("  VALIDATION COMPLETE!")
    print("="*70)
    print(f"\nAll files saved to: {WORK_DIR}")
    print("\nGenerated Files:")
    print("  📊 classification_report.txt     - Detailed per-class metrics")
    print("  📈 confusion_matrix.png          - Confusion matrix heatmap")
    print("  📉 per_class_metrics.png         - Per-class performance chart")
    print("  📁 best_model_fold1.pth          - Model from fold 1")
    print("  📁 best_model_fold2.pth          - Model from fold 2")
    print("  📁 best_model_fold3.pth          - Model from fold 3")
    print("  📁 best_model_fold4.pth          - Model from fold 4")
    print("  📁 best_model_fold5.pth          - Model from fold 5")
    print("  🔧 best_model_finetuned.pth      - Fine-tuned model")
    print("  📋 validation_summary.json        - Machine-readable summary")
    print("\n" + "-"*70)
    print("Ready for your project report!")
    print("="*70 + "\n")
    print("\nFinal folder contents:")
    for item in sorted(os.listdir(WORK_DIR)):
        size = os.path.getsize(os.path.join(WORK_DIR, item))
        print(f"  - {item} ({size/1024:.1f} KB)")
if __name__ == "__main__":
    main()

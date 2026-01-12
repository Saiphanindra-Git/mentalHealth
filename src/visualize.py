import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from models.bert_classifier import MentalHealthBERTClassifier
from models.dataset import MentalHealthDataset

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_model_and_data(device):
    """Load trained model and test dataset"""
    print("Loading model and data...")
    
    # Load checkpoint
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Load test dataset
    test_dataset = MentalHealthDataset(
        PROCESSED_TEST_FILE,
        tokenizer,
        max_length=MAX_LENGTH,
        label_encoder=checkpoint['label_encoder']
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load model with correct dropout from config
    dropout_rate = DROPOUT if 'DROPOUT' in dir() else 0.4
    model = MentalHealthBERTClassifier(
        model_name=MODEL_NAME, 
        num_labels=NUM_LABELS, 
        dropout=dropout_rate
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, test_loader, test_dataset, checkpoint

def get_predictions(model, dataloader, device):
    """Get predictions on test set"""
    print("Generating predictions...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_training_history():
    """Plot training curves"""
    print("\n1. Plotting training history...")
    
    history_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Progress - 6 Epochs', fontsize=16, fontweight='bold', y=1.02)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', 
                 linewidth=2, markersize=8, color='#2E86AB')
    axes[0].plot(epochs, history['val_loss'], 's-', label='Val Loss', 
                 linewidth=2, markersize=8, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    
    # Accuracy
    axes[1].plot(epochs, [acc*100 for acc in history['train_acc']], 'o-', 
                 label='Train Accuracy', linewidth=2, markersize=8, color='#2E86AB')
    axes[1].plot(epochs, [acc*100 for acc in history['val_acc']], 's-', 
                 label='Val Accuracy', linewidth=2, markersize=8, color='#A23B72')
    axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Target')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(50, 100)
    
    # F1 Score
    axes[2].plot(epochs, [f1*100 for f1 in history['train_f1']], 'o-', 
                 label='Train F1', linewidth=2, markersize=8, color='#2E86AB')
    axes[2].plot(epochs, [f1*100 for f1 in history['val_f1']], 's-', 
                 label='Val F1', linewidth=2, markersize=8, color='#A23B72')
    axes[2].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Target')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Training & Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11, loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(50, 100)
    
    # Mark best F1 epoch
    best_f1_epoch = np.argmax(history['val_f1']) + 1
    axes[2].axvline(x=best_f1_epoch, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    print("\n2. Plotting confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Confusion Matrix - Test Set Performance', fontsize=16, fontweight='bold', y=0.98)
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                yticklabels=labels, ax=axes[0], cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', xticklabels=labels, 
                yticklabels=labels, ax=axes[1], cbar_kws={'label': 'Proportion'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'}, vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()

def plot_per_class_performance(y_true, y_pred, labels):
    """Plot per-class metrics"""
    print("\n3. Plotting per-class performance...")
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels))
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Sort by F1-Score for better visualization
    df_sorted = df.sort_values('F1-Score', ascending=True)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
    
    # Precision
    colors_prec = ['lightcoral' if x < 0.7 else 'skyblue' for x in df_sorted['Precision']]
    axes[0, 0].barh(df_sorted['Class'], df_sorted['Precision'], color=colors_prec, edgecolor='black')
    axes[0, 0].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% Target')
    axes[0, 0].set_xlabel('Precision', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim(0, 1.05)
    axes[0, 0].legend()
    for i, v in enumerate(df_sorted['Precision']):
        axes[0, 0].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # Recall
    colors_rec = ['lightcoral' if x < 0.7 else 'lightyellow' if x < 0.85 else 'lightgreen' for x in df_sorted['Recall']]
    axes[0, 1].barh(df_sorted['Class'], df_sorted['Recall'], color=colors_rec, edgecolor='black')
    axes[0, 1].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% Target')
    axes[0, 1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlim(0, 1.05)
    axes[0, 1].legend()
    for i, v in enumerate(df_sorted['Recall']):
        axes[0, 1].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # F1 Score
    colors_f1 = ['lightcoral' if x < 0.7 else 'lightyellow' if x < 0.85 else 'lightgreen' for x in df_sorted['F1-Score']]
    axes[1, 0].barh(df_sorted['Class'], df_sorted['F1-Score'], color=colors_f1, edgecolor='black')
    axes[1, 0].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% Target')
    axes[1, 0].set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('F1-Score by Class (Sorted)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlim(0, 1.05)
    axes[1, 0].legend()
    for i, v in enumerate(df_sorted['F1-Score']):
        axes[1, 0].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # Support
    axes[1, 1].barh(df['Class'], df['Support'], color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Support (Test Samples) by Class', fontsize=14, fontweight='bold')
    for i, v in enumerate(df['Support']):
        axes[1, 1].text(v + 50, i, f'{int(v)}', va='center', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()
    
    return df

def plot_prediction_confidence(y_true, y_pred, probs, labels):
    """Plot prediction confidence distribution"""
    print("\n4. Plotting prediction confidence...")
    
    # Get confidence scores (max probability)
    confidences = np.max(probs, axis=1)
    correct = (y_true == y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Prediction Confidence Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Confidence distribution
    axes[0].hist(confidences[correct], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(confidences[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Add stats
    correct_conf = confidences[correct].mean()
    incorrect_conf = confidences[~correct].mean()
    axes[0].text(0.05, 0.95, f'Correct avg: {correct_conf:.3f}\nIncorrect avg: {incorrect_conf:.3f}',
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Confidence by class
    class_confidences = []
    for i, label in enumerate(labels):
        mask = (y_true == i)
        if mask.sum() > 0:
            class_confidences.append(confidences[mask].mean())
        else:
            class_confidences.append(0)
    
    colors = ['lightcoral' if c < 0.85 else 'lightyellow' if c < 0.92 else 'lightgreen' for c in class_confidences]
    axes[1].bar(labels, class_confidences, color=colors, edgecolor='black', alpha=0.8)
    axes[1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% Confidence')
    axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
    axes[1].set_title('Average Prediction Confidence by Class', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].set_ylim(0.7, 1.0)
    for i, v in enumerate(class_confidences):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'confidence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()

def save_detailed_report(y_true, y_pred, labels, checkpoint):
    """Save detailed classification report"""
    print("\n5. Generating detailed report...")
    
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MENTAL HEALTH CLASSIFICATION - TEST SET RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Best Epoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Validation F1: {checkpoint['val_f1']:.4f} ({checkpoint['val_f1']*100:.2f}%)\n")
        f.write(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}\n")
        f.write(f"Max Sequence Length: {MAX_LENGTH} tokens\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Dropout: {DROPOUT if 'DROPOUT' in dir() else 0.4}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    
    print(f"   ✓ Saved: {report_path}")

def main():
    print("="*70)
    print("MODEL EVALUATION & VISUALIZATION")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and data
    model, test_loader, test_dataset, checkpoint = load_model_and_data(device)
    labels = test_dataset.get_labels()
    
    print(f"\nModel Configuration:")
    print(f"  Checkpoint from Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Validation F1: {checkpoint['val_f1']:.4f} ({checkpoint['val_f1']*100:.2f}%)")
    
    # Get predictions
    y_pred, y_true, probs = get_predictions(model, test_loader, device)
    
    # Calculate overall metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n{'='*70}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 (Weighted):   {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"F1 (Macro):      {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    
    if f1_weighted >= 0.90:
        print(f"\n🎉 TARGET ACHIEVED: 90%+ F1 Score!")
    else:
        print(f"\nGap to 90% target: {(0.90 - f1_weighted)*100:.2f}%")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_training_history()
    plot_confusion_matrix(y_true, y_pred, labels)
    per_class_df = plot_per_class_performance(y_true, y_pred, labels)
    plot_prediction_confidence(y_true, y_pred, probs, labels)
    save_detailed_report(y_true, y_pred, labels, checkpoint)
    
    # Save per-class metrics
    per_class_df.to_csv(os.path.join(RESULTS_DIR, 'per_class_metrics.csv'), index=False)
    
    print(f"\n{'='*70}")
    print("✅ VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll plots saved to: {os.path.join(RESULTS_DIR, 'plots')}")
    print(f"\nGenerated files:")
    print(f"  1. training_curves.png (updated with 6 epochs)")
    print(f"  2. confusion_matrix.png (updated)")
    print(f"  3. per_class_metrics.png (updated)")
    print(f"  4. confidence_analysis.png (updated)")
    print(f"  5. classification_report.txt (updated)")
    print(f"  6. per_class_metrics.csv (updated)")
    print(f"\n📊 All visualizations have been updated with latest model!")

if __name__ == "__main__":
    main()

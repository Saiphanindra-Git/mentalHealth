import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import json

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from models.bert_classifier import MentalHealthBERTClassifier
from models.dataset import MentalHealthDataset

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================
# LABEL SMOOTHING LOSS FOR BETTER GENERALIZATION
# ============================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing regularization"""
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            
            # Apply class weights if provided
            if self.weight is not None:
                true_dist = true_dist * self.weight.unsqueeze(0)
                true_dist = true_dist / true_dist.sum(dim=1, keepdim=True)
        
        loss = torch.mean(torch.sum(-true_dist * log_pred, dim=-1))
        return loss

def calculate_class_weights(train_dataset):
    """Calculate class weights for imbalanced dataset"""
    train_data = pd.read_csv(PROCESSED_TRAIN_FILE)
    label_counts = train_data['label'].value_counts()
    
    total_samples = len(train_data)
    num_classes = len(label_counts)
    
    # Calculate inverse frequency weights
    weights = []
    for label in train_dataset.get_labels():
        count = label_counts[label]
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    
    print(f"\n⚖️  Class Weights (handling 17.91:1 imbalance):")
    for label, weight in zip(train_dataset.get_labels(), weights):
        print(f"   {label:25s}: {weight:.4f}")
    
    return weights

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, gradient_accum_steps=2):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc="Training")
    
    for idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accum_steps
        loss.backward()
        
        # Update weights every gradient_accum_steps
        if (idx + 1) % gradient_accum_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * gradient_accum_steps
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item() * gradient_accum_steps})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def load_checkpoint_if_exists(model, optimizer, device):
    """Load checkpoint if exists, return start_epoch and best metrics"""
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found existing checkpoint!")
        response = input("   Resume training from checkpoint? (y/n): ").lower()
        
        if response == 'y':
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move model to device first
            model = model.to(device)
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Move optimizer states to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['val_f1']
            label_encoder = checkpoint['label_encoder']
            
            print(f"   ✓ Resuming from Epoch {start_epoch}")
            print(f"   ✓ Previous best Val F1: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
            
            # Load history
            history_path = os.path.join(RESULTS_DIR, 'training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
            else:
                history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                          'val_loss': [], 'val_acc': [], 'val_f1': []}
            
            return model, start_epoch, best_val_f1, label_encoder, history, checkpoint['epoch']
    
    return None, 0, 0, None, None, -1

def main():
    print("="*70)
    print("MENTAL HEALTH BERT CLASSIFIER - COMPLETE TRAINING")
    print("="*70)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   ⚡ Training will be ~15x faster than CPU!")
    else:
        print(f"   ⚠️  Running on CPU (will be slower)")
    
    # Load tokenizer
    print(f"\n📚 Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Initialize model first for checkpoint loading
    print(f"\n🤖 Initializing BERT model...")
    model = MentalHealthBERTClassifier(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        dropout=DROPOUT if 'DROPOUT' in dir() else 0.4
    )
    
    # Initialize optimizer for checkpoint loading
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        eps=1e-8,
        weight_decay=WEIGHT_DECAY if 'WEIGHT_DECAY' in dir() else 0.01
    )
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint_if_exists(model, optimizer, device)
    
    if checkpoint_data[0] is not None:
        model, start_epoch, best_val_f1, label_encoder_ckpt, history, best_epoch = checkpoint_data
        resume_mode = True
    else:
        model = model.to(device)
        start_epoch = 0
        best_val_f1 = 0
        label_encoder_ckpt = None
        history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                  'val_loss': [], 'val_acc': [], 'val_f1': []}
        best_epoch = 0
        resume_mode = False
        print("   Starting fresh training...")
    
    # Create datasets
    print(f"\n📊 Loading datasets...")
    if resume_mode and label_encoder_ckpt:
        train_dataset = MentalHealthDataset(
            PROCESSED_TRAIN_FILE, 
            tokenizer, 
            max_length=MAX_LENGTH,
            label_encoder=label_encoder_ckpt
        )
        val_dataset = MentalHealthDataset(
            PROCESSED_VAL_FILE,
            tokenizer,
            max_length=MAX_LENGTH,
            label_encoder=label_encoder_ckpt
        )
    else:
        train_dataset = MentalHealthDataset(
            PROCESSED_TRAIN_FILE, 
            tokenizer, 
            max_length=MAX_LENGTH
        )
        val_dataset = MentalHealthDataset(
            PROCESSED_VAL_FILE,
            tokenizer,
            max_length=MAX_LENGTH,
            label_encoder=train_dataset.get_label_encoder()
        )
    
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Number of classes: {NUM_LABELS}")
    print(f"   Classes: {train_dataset.get_labels()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Loss function with label smoothing
    label_smoothing = LABEL_SMOOTHING if 'LABEL_SMOOTHING' in dir() else 0.1
    criterion = LabelSmoothingCrossEntropy(weight=class_weights, smoothing=label_smoothing)
    
    # Reinitialize optimizer if not resuming
    if not resume_mode:
        optimizer = AdamW(
            model.parameters(), 
            lr=LEARNING_RATE, 
            eps=1e-8,
            weight_decay=WEIGHT_DECAY if 'WEIGHT_DECAY' in dir() else 0.01
        )
    
    # Learning rate scheduler
    gradient_accum_steps = GRADIENT_ACCUMULATION_STEPS if 'GRADIENT_ACCUMULATION_STEPS' in dir() else 2
    remaining_epochs = EPOCHS - start_epoch
    total_steps = (len(train_loader) // gradient_accum_steps) * remaining_epochs
    warmup_ratio = WARMUP_RATIO if 'WARMUP_RATIO' in dir() else 0.15
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training configuration
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model:                      {MODEL_NAME}")
    print(f"Max sequence length:        {MAX_LENGTH} tokens")
    print(f"Batch size:                 {BATCH_SIZE}")
    print(f"Effective batch size:       {BATCH_SIZE * gradient_accum_steps}")
    print(f"Learning rate:              {LEARNING_RATE}")
    print(f"Dropout:                    {DROPOUT if 'DROPOUT' in dir() else 0.4}")
    print(f"Label smoothing:            {label_smoothing}")
    print(f"Weight decay:               {WEIGHT_DECAY if 'WEIGHT_DECAY' in dir() else 0.01}")
    print(f"Epochs:                     {EPOCHS}")
    print(f"Starting from epoch:        {start_epoch + 1}")
    print(f"Warmup steps:               {warmup_steps}")
    print(f"Gradient accumulation:      {gradient_accum_steps} steps")
    
    if 'PATIENCE' in dir():
        print(f"Early stopping patience:    {PATIENCE} epochs")
    
    print(f"{'='*70}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    
    patience_counter = 0
    patience = PATIENCE if 'PATIENCE' in dir() else 3
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, gradient_accum_steps
        )
        
        # Validate
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%), F1: {train_f1:.4f} ({train_f1*100:.2f}%)")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%), F1: {val_f1:.4f} ({val_f1*100:.2f}%)")
        
        # Save best model
        if val_f1 > best_val_f1:
            improvement = (val_f1 - best_val_f1) * 100
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Create models directory
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save model
            model_path = os.path.join(MODEL_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'label_encoder': train_dataset.get_label_encoder()
            }, model_path)
            
            print(f"\n✅ New best model saved! F1: {val_f1:.4f} ({val_f1*100:.2f}%)")
            if epoch > 0:
                print(f"   Improvement: +{improvement:.2f}%")
            
            if val_f1 >= 0.90:
                print(f"   🎉 TARGET ACHIEVED: 90%+ ACCURACY!")
        else:
            patience_counter += 1
            print(f"\n   No improvement. Patience: {patience_counter}/{patience}")
            print(f"   Current best: Epoch {best_epoch}, F1: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n🎯 Final Results:")
    print(f"   Best Epoch:              {best_epoch}")
    print(f"   Best Validation F1:      {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    print(f"   Best Validation Acc:     {history['val_acc'][best_epoch-1]:.4f} ({history['val_acc'][best_epoch-1]*100:.2f}%)")
    print(f"   Model saved to:          {MODEL_DIR}/best_model.pt")
    print(f"   Training history:        {history_path}")
    
    if best_val_f1 >= 0.90:
        print(f"\n🎉🎉🎉 GOAL ACHIEVED: 90%+ ACCURACY! 🎉🎉🎉")
    elif best_val_f1 >= 0.85:
        print(f"\n✨ Great performance! Gap to 90%: {(0.90 - best_val_f1)*100:.2f}%")
    else:
        print(f"\n   Gap to 90% target: {(0.90 - best_val_f1)*100:.2f}%")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run: python src\\visualize.py (to visualize results)")
    print(f"   2. Run: python src\\evaluate.py (to test on held-out test set)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

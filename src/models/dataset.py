import torch
from torch.utils.data import Dataset
import pandas as pd

class MentalHealthDataset(Dataset):
    """Custom Dataset for mental health text classification"""
    
    def __init__(self, csv_file, tokenizer, max_length=128, label_encoder=None):
        """
        Args:
            csv_file: Path to CSV file
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            label_encoder: Dictionary mapping labels to indices
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label encoder if not provided
        if label_encoder is None:
            self.labels = sorted(self.data['label'].unique())
            self.label_encoder = {label: idx for idx, label in enumerate(self.labels)}
        else:
            self.label_encoder = label_encoder
            self.labels = [label for label, _ in sorted(label_encoder.items(), key=lambda x: x[1])]
        
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single item"""
        text = str(self.data.iloc[idx]['text'])
        label = self.data.iloc[idx]['label']
        label_idx = self.label_encoder[label]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_idx, dtype=torch.long)
        }
    
    def get_labels(self):
        """Return list of label names"""
        return self.labels
    
    def get_label_encoder(self):
        """Return label encoder dictionary"""
        return self.label_encoder

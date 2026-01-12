import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MentalHealthBERTClassifier(nn.Module):
    """BERT-based classifier for mental health text classification"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=7, dropout=0.3):
        super(MentalHealthBERTClassifier, self).__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        Returns:
            logits: Classification scores (batch_size, num_labels)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        return logits

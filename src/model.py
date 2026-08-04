import torch
import torch.nn as nn

class CustomTextClassifier(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=128, num_classes=24):
        """
        Built entirely from scratch. 
        Total Parameters: ~3.8 Million (Well under the 5B limit)
        """
        super(CustomTextClassifier, self).__init__()
        
        # Embedding Layer (No pretrained weights)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # Global Average Pooling (Extracts sequence meaning efficiently)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)                     # (batch_size, seq_len, embed_dim)
        
        # Permute for pooling layer: (batch_size, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)             
        
        # Pool across the sequence length
        pooled = self.pool(embedded).squeeze(2)          # (batch_size, embed_dim)
        
        # Classification Head
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        logits = self.out(x)                             # (batch_size, num_classes)
        
        return logits

# Quick test to verify parameter count
if __name__ == "__main__":
    model = CustomTextClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model successfully initialized!")
    print(f"Total Parameters: {total_params:,}")
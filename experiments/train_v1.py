import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from model import CustomTextClassifier
import time
import json

# 1. Hardware Auto-Detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Hardware Detected: {device} ---")

# 2. Custom Streaming Dataset (Prevents Out-Of-Memory Errors)
class StreamingParquetDataset(IterableDataset):
    def __init__(self, parquet_file, tokenizer_path, max_len=256):
        self.parquet_file = parquet_file
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.topic_to_id = {}
        
    def __iter__(self):
        # Stream the parquet file in optimized row groups
        pf = pq.ParquetFile(self.parquet_file)
        for batch in pf.iter_batches(batch_size=2048):
            # Convert pyarrow batch to python dictionaries
            for row in batch.to_pylist():
                text = str(row['DATA']) if row['DATA'] else ""
                topic = str(row['TOPIC'])
                
                # Dynamic Label Encoding
                if topic not in self.topic_to_id:
                    self.topic_to_id[topic] = len(self.topic_to_id)
                label_id = self.topic_to_id[topic]
                
                # Tokenization & Padding
                encoded = self.tokenizer.encode(text)
                input_ids = encoded.ids[:self.max_len]
                
                if len(input_ids) < self.max_len:
                    input_ids += [0] * (self.max_len - len(input_ids))
                    
                yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)

def train():
    print("Initializing Streaming Pipeline...")
    # Initialize Dataset and DataLoader
    dataset = StreamingParquetDataset("dataset_10M.parquet", "final_models/custom_tokenizer.json")
    dataloader = DataLoader(dataset, batch_size=128)
    
    # Initialize Model, Loss, and Optimizer
    model = CustomTextClassifier(vocab_size=30000, num_classes=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Using standard CrossEntropyLoss (you can add class weights here later if needed)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training Loop...")
    model.train()
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward Pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Time Elapsed: {time.time() - start_time:.2f}s")
            
        
    print("\n✅ Pipeline Test Complete! Model is learning.")
    print("To train on the full dataset, remove the 'break' statement in train.py.")
    
    # Save the dynamically generated topic mapping
    with open("experiments/topic_mapping_v1.json", "w") as f:
        json.dump(dataset.topic_to_id, f)
    print("Saved topic_mapping.json")
    
    # Save final model weights
    torch.save(model.state_dict(), "experiments/final_model.pth")
    print("Saved final_model.pth")
            

if __name__ == "__main__":
    train()
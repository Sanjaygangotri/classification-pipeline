import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import polars as pl
from tokenizers import Tokenizer
from model import CustomTextClassifier
import time
import json
import os
import random
import numpy as np

def set_seed(seed=42):
    """Locks all random operations to ensure reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Hardware Detected: {device} ---")

class StreamingParquetDataset(IterableDataset):
    def __init__(self, parquet_file, tokenizer_path, topic_mapping, max_len=256):
        self.parquet_file = parquet_file
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.topic_to_id = topic_mapping
        
    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_file)
        for batch in pf.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                text = str(row['DATA']) if row['DATA'] else ""
                topic = str(row['TOPIC'])
                
                # Skip unknown topics just in case
                if topic not in self.topic_to_id:
                    continue
                    
                label_id = self.topic_to_id[topic]
                
                encoded = self.tokenizer.encode(text)
                input_ids = encoded.ids[:self.max_len]
                
                if len(input_ids) < self.max_len:
                    input_ids += [0] * (self.max_len - len(input_ids))
                    
                yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)

def compute_class_weights(dataset_path, topic_mapping):
    print("Computing dynamic class weights using Polars...")
    lazy_df = pl.scan_parquet(dataset_path)
    
    # Count frequencies of each topic
    counts = lazy_df.group_by("TOPIC").agg(pl.len().alias("count")).collect()
    total_samples = counts["count"].sum()
    num_classes = len(topic_mapping)
    
    weights = torch.ones(num_classes)
    
    # Apply inverse frequency formula: w_i = N / (C * n_i)
    for row in counts.iter_rows(named=True):
        topic = row["TOPIC"]
        count = row["count"]
        if topic in topic_mapping:
            label_id = topic_mapping[topic]
            weights[label_id] = total_samples / (num_classes * count)
            
    print("Class weights successfully computed!")
    return weights.to(device)

def train2():
    print("Initializing Experiment 2: Weighted Loss Pipeline...")
    
    # Load the mapping generated from the first run
    if not os.path.exists("final_models/topic_mapping.json"):
        raise FileNotFoundError("topic_mapping.json not found. Run train.py first.")
        
    with open("final_models/topic_mapping.json", "r") as f:
        topic_mapping = json.load(f)
    
    # Compute weights before starting the loop
    class_weights = compute_class_weights("dataset_10M.parquet", topic_mapping)
    
    dataset = StreamingParquetDataset("dataset_10M.parquet", "final_models/custom_tokenizer.json", topic_mapping)
    dataloader = DataLoader(dataset, batch_size=128)
    
    model = CustomTextClassifier(vocab_size=30000, num_classes=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Injecting the weights into the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print("Starting Weighted Training Loop...")
    model.train()
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Time Elapsed: {time.time() - start_time:.2f}s")
            
    print("\n✅ Experiment 2 Training Complete!")
    
    # Save as a new file to protect your baseline model
    torch.save(model.state_dict(), "final_models/final_model.pth")
    print("Saved final_models/final_model.pth")

if __name__ == "__main__":
    train2()
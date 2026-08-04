import torch
import json
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from model import CustomTextClassifier
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

def evaluate_model_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running V2 Evaluation on: {device} ---")

    # Load mappings and tokenizer
    try:
        with open("final_models/topic_mapping.json", "r") as f:
            topic_to_id = json.load(f)
    except FileNotFoundError:
        print("Error: topic_mapping.json not found.")
        return
        
    id_to_topic = {v: k for k, v in topic_to_id.items()}
    tokenizer = Tokenizer.from_file("final_models/custom_tokenizer.json")
    max_len = 256
    
    # Initialize model and load V2 weights
    print("Loading final_model_v2.pth weights...")
    model = CustomTextClassifier(vocab_size=30000, num_classes=24).to(device)
    model.load_state_dict(torch.load("final_models/final_model.pth", map_location=device, weights_only=True))
    model.eval()

    y_true = []
    y_pred = []
    
    print("Evaluating on a 10,000-row sample from the dataset...")
    pf = pq.ParquetFile("dataset_10M.parquet")
    
    with torch.no_grad():
        for batch in pf.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                text = str(row['DATA']) if row['DATA'] else ""
                topic = str(row['TOPIC'])
                
                if topic not in topic_to_id:
                    continue
                    
                label_id = topic_to_id[topic]
                encoded = tokenizer.encode(text)
                input_ids = encoded.ids[:max_len]
                
                if len(input_ids) < max_len:
                    input_ids += [0] * (max_len - len(input_ids))
                
                inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                outputs = model(inputs)
                predicted_id = torch.argmax(outputs, dim=1).item()
                
                y_true.append(label_id)
                y_pred.append(predicted_id)
            
            if len(y_true) >= 10000:
                break

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print("\n=== FINAL EVALUATION METRICS (V2) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    # Generate target names in the correct ID order
    target_names = [id_to_topic[i] for i in range(len(id_to_topic))]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    evaluate_model_v2()
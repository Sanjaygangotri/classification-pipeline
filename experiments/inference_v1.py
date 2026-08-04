import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


import torch
import json
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import CustomTextClassifier

def evaluate_model():
    # Detect Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Evaluation on: {device} ---")

    # Load the topic mapping generated during training
    try:
        with open("final_models/topic_mapping.json", "r") as f:
            topic_to_id = json.load(f)
    except FileNotFoundError:
        print("Error: topic_mapping.json not found. Please complete training first.")
        return

    # Load custom tokenizer
    tokenizer = Tokenizer.from_file("final_models/custom_tokenizer.json")
    max_len = 256
    
    # Initialize the model and load weights
    print("Loading model weights...")
    model = CustomTextClassifier(vocab_size=30000, num_classes=24).to(device)
    model.load_state_dict(torch.load("experiments/final_model.pth", map_location=device, weights_only=True))
    model.eval()

    y_true = []
    y_pred = []
    
    print("Evaluating on a 10,000-row sample from the dataset...")
    pf = pq.ParquetFile("dataset_10M.parquet")
    
    # Run evaluation without computing gradients to save memory
    with torch.no_grad():
        for batch in pf.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                text = str(row['DATA']) if row['DATA'] else ""
                topic = str(row['TOPIC'])
                
                # Skip unmapped topics (if any)
                if topic not in topic_to_id:
                    continue
                    
                label_id = topic_to_id[topic]
                
                # Tokenize exactly as done in training
                encoded = tokenizer.encode(text)
                input_ids = encoded.ids[:max_len]
                
                if len(input_ids) < max_len:
                    input_ids += [0] * (max_len - len(input_ids))
                
                inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                # Predict
                outputs = model(inputs)
                predicted_id = torch.argmax(outputs, dim=1).item()
                
                y_true.append(label_id)
                y_pred.append(predicted_id)
            
            # Stop after 10,000 rows to keep evaluation fast
            if len(y_true) >= 10000:
                break

    # Calculate required Evaluation Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print("\n=== FINAL EVALUATION METRICS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate_model()
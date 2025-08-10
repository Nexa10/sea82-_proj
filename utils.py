import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test_google_colab():
  print("testing...")

def evaluate_model_performance(all_true_labels, all_predictions, model_name="Model"):
    """
    Calculates and displays accuracy, classification report, confusion matrix
    """

    # Calculate accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
  
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions,
                              target_names=['Human-written', 'AI-generated']))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true_labels, all_predictions)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".4f", cmap="Blues",
                xticklabels=['Human (0)', 'AI (1)'],
                yticklabels=['Human (0)', 'AI (1)'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix (Normalized)")
    plt.show()

    # Print summary stats
    total_samples = len(all_true_labels)
    correct_predictions = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    print(f"\nSummary:")
    print(f"Total test samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")


# setup device for training
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# Enhanced model saving to handle UnicodeDecodeError and ensure proper LoRA adapter persistence
def save_model(model, tokenizer, save_path="./saved_model/distilbert_lora_final", 
               model_info=None):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate model state before saving
        if not hasattr(model, 'save_pretrained'):
            raise ValueError("No save_pretrained method.")
        
        # Save LoRA adapter weights and configuration
        print("Saving LoRA adapter weights and config...")
        model.save_pretrained(save_path)
        
        # Save tokenizer
        print("Saving tokenizer...")
        tokenizer.save_pretrained(save_path)
        
        # Save comprehensive model metadata
        if model_info is None:
            model_info = {
                "base_model": "distilbert-base-uncased",
                "model_type": "distilbert_lora",
                "task": "sequence_classification",
                "num_labels": 2,
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"],
                    "lora_dropout": 0.1
                }
            }
        
        with open(save_path / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model saved to {save_path}")
        return True

    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False


def load_model(model_path="./saved_model/distilbert_lora_final", device=None):
    if device is None:
        device = setup_device()
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return None, None
    
    try:    
        print(f"Loading DistilBERT + LoRA model from {model_path}")
        
        # Load model info if available
        info_path = model_path / "model_info.json"
        base_model_name = "distilbert-base-uncased"
        num_labels = 2
        
        if info_path.exists():
            with open(info_path, "r") as f:
                model_info = json.load(f)
                base_model_name = model_info.get("base_model", "distilbert-base-uncased")
                num_labels = model_info.get("num_labels", 2)
                print(f"Using base model: {base_model_name}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"Failed to load tokenizer from model path, using base model tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        print(f"Loading base DistilBERT model...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, 
            num_labels=num_labels
        )
        
        # Load LoRA adapter
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
        model = model.to(device)
        
        print(f"Model loaded")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

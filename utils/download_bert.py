from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import argparse

def download_and_save_bert(save_path):
    """Download BERT model and tokenizer and save them to the specified path"""
    print(f"Downloading BERT model and tokenizer to {save_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(save_path)
    print("Tokenizer saved successfully!")
    
    # Download and save model
    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    model.save_pretrained(save_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and save BERT model')
    parser.add_argument('--save_path', type=str, required=True,
                      help='Path to save the BERT model and tokenizer')
    
    args = parser.parse_args()
    download_and_save_bert(args.save_path)
    print("Done!")

import pandas as pd
import requests
import os
from datetime import datetime
import random
import urllib3
import spacy
from collections import Counter
import re
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# API configuration
API_KEY = ""

def extract_frequent_nouns(utterances, top_n=5):
    """Extract the most frequent nouns from a list of utterances using spaCy"""
    # Combine all utterances into one text
    all_nouns = []
    
    for utterance in utterances:
        doc = nlp(utterance)
        # Extract nouns and compound nouns
        for chunk in doc.noun_chunks:
            all_nouns.append(chunk.text.lower())
        # Also add single nouns
        nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
        all_nouns.extend(nouns)
    
    # Count frequencies
    noun_freq = Counter(all_nouns)
    # Get top N most common nouns
    most_common = noun_freq.most_common(top_n)
    return [noun for noun, _ in most_common]

def generate_label_prompt(utterances):
    # Extract frequent nouns
    frequent_nouns = extract_frequent_nouns(utterances)
    frequent_words = ", ".join(frequent_nouns)
    
    prompt = f"""
    Generate a single user intent label in the format: verb_noun(s).
    Rules:
    1. Start with an action verb (e.g., show, get, find, search)
    2. Add underscore (_) after the verb
    3. End with the most relevant noun(s) from the utterances
    4. Return ONLY the label, nothing else

    Example:
    Input: "What is the H-index of persons who have authored a publication from 2024 onward?"
    Output: show_h_index

    Utterances to analyze: {utterances}
    """
    
#     prompt = f"""
# Generate a single user intent label by several words as precise as possible
# Return ONLY the label, nothing else
# For example: Text:How can I lookup where funds came from? label:verify_source_of_funds
# Utterances to analyze: {utterances}
# """
    return prompt

def clean_generated_label(text):
    """Extract only words containing underscores from the generated text"""
    # Find all words containing underscore
    matches = re.findall(r'\b\w+_\w+(?:_\w+)*\b', text)
    
    if matches:
        # Return the last match (in case there are multiple)
        return matches[-1].strip()
    return text.strip()  # Return original text if no underscore words found

def call_api(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates user intent labels and answer by exactly one word in the format action(verb)_nouns"},
            #{"role": "system", "content": "You are a helpful assistant that generates user intent label"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        print("\nMaking API request with prompt:", prompt)
        response = requests.post(url, headers=headers, json=data, verify=False)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"Raw Generated Text: {generated_text}")
        
        # Clean and filter the generated text
        cleaned_label = clean_generated_label(generated_text)
        print(f"Cleaned Label: {cleaned_label}\n")
        return cleaned_label
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Response content: {response.text}")
        return None

def process_dataset(input_file, output_file):
    # Read the input TSV file
    df = pd.read_csv(input_file, sep='\t')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a new column for generated labels if it doesn't exist
    if 'gen_label' not in df.columns:
        df['gen_label'] = None
    
    # Save initial version of the file
    df.to_csv(output_file, sep='\t', index=False)
    
    # 修改：使用 predicted_label_num 进行分组
    grouped = df.groupby('predicted_label_num')
    total_groups = len(df['predicted_label_num'].unique())
    processed_count = 0
    
    # Process each label group
    for pred_num, group in grouped:  # 修改变量名以更好地反映内容
        processed_count += 1
        print(f"\nProcessing prediction group {processed_count}/{total_groups}: {pred_num}")
        
        # Convert group texts to list
        texts = group['text'].tolist()
        
        # Randomly select 3 utterances (or all if less than 3) to set
        sample_size = min(19, len(texts))
        sample_utterances = random.sample(texts, sample_size)
        
        # Generate prompt with the sample utterances
        prompt = generate_label_prompt(sample_utterances)
        
        # Call API to get the generated label
        generated_label = call_api(prompt)
        
        if generated_label:
            # 修改：更新条件使用 predicted_label_num
            df.loc[df['predicted_label_num'] == pred_num, 'gen_label'] = generated_label
            # Save the updated results immediately
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Updated results saved to {output_file}")
        else:
            print(f"Warning: No label generated for prediction group {pred_num}")
    
    print(f"\nProcessing completed. Final results saved to {output_file}")
    print(f"Processed {processed_count} prediction groups")

if __name__ == "__main__":
    input_file = "dataset/prediction_errors.tsv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/prediction_errors_with_generated_labels_bydeepseek_{timestamp}.tsv"
    
    process_dataset(input_file, output_file)

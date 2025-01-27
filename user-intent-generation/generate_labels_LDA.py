import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import random
from datetime import datetime
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess the text"""
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

def extract_verbs_and_nouns(texts):
    """Extract verbs and nouns"""
    all_words = []
    for text in texts:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        # Get verbs (VB*) and nouns (NN*)
        verbs = [word.lower() for word, tag in tagged if tag.startswith('VB')]
        nouns = [word.lower() for word, tag in tagged if tag.startswith('NN')]
        all_words.extend(verbs + nouns)
    return verbs, nouns

def get_core_nouns(text):
    """Extract core nouns using TF-IDF for importance scoring but keep original mapping"""
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        tokenizer=word_tokenize,
        stop_words=None,  # Don't use built-in stop words
        ngram_range=(1, 1)
    )
    
    # Fit and transform the single text (wrapped in a list)
    tfidf_matrix = tfidf.fit_transform([text])
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    
    # Keep original ignore words and mapping
    ignore_words = {'what', 'is', 'are', 'the', 'a', 'an', 'by', 'in', 'at', 'of', 
                   'with', 'to', 'from', 'who', 'how', 'many', 'can', 'you', 'me', 
                   'tell', 'give', 'show', 'provide', 'share'}
    
    core_noun_mapping = {
        'paper': {'paper', 'publication', 'work', 'research'},
        'author': {'author', 'person', 'researcher', 'figure'},
        'coauthor': {'coauthor', 'co-author'},
        'venue': {'venue', 'location', 'place'},
        'abstract': {'abstract'},
        'doi': {'doi', 'identifier'},
        'type': {'type', 'genre', 'category', 'classification', 'nature'},
        'citation': {'citation', 'reference', 'cited'}
    }
    
    # Get POS tagged tokens
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    
    # Extract nouns and filter stop words
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    nouns = [n for n in nouns if n not in ignore_words]
    
    # Sort nouns by TF-IDF score
    nouns.sort(key=lambda x: tfidf_scores.get(x, 0), reverse=True)
    
    # Keep original mapping logic but with sorted nouns
    core_nouns = []
    used_categories = set()
    
    # First try to map to core categories
    for noun in nouns:
        for category, synonyms in core_noun_mapping.items():
            if noun in synonyms and category not in used_categories:
                core_nouns.append(category)
                used_categories.add(category)
                break
    
    # Add remaining nouns that weren't mapped
    remaining_nouns = [n for n in nouns if not any(n in s for s in core_noun_mapping.values())]
    core_nouns.extend(remaining_nouns)
    
    return core_nouns

def generate_intent_label(texts):
    """Generate intent label"""
    all_core_nouns = []
    
    # Extract core nouns from all sample texts
    for text in texts:
        core_nouns = get_core_nouns(text)
        all_core_nouns.extend(core_nouns)
    
    # Count noun frequencies
    noun_counts = pd.Series(all_core_nouns).value_counts()
    
    # Select up to two most frequent core nouns
    selected_nouns = noun_counts.head(2).index.tolist()
    
    # If no nouns found, use default value
    if not selected_nouns:
        selected_nouns = ['information']
    
    # Generate label (show + core nouns)
    #intent_label = f"show_{'_'.join(selected_nouns)}"
    intent_label = f"{'_'.join(selected_nouns)}"
    return intent_label

def process_dataset(input_file, output_file):
    """Process dataset and generate labels"""
    # Read input file
    df = pd.read_csv(input_file, sep='\t')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Add generated label column
    df['gen_label'] = None
    
    # Process by predicted_label_num
    grouped = df.groupby('predicted_label_num')
    total_groups = len(df['predicted_label_num'].unique())
    
    for idx, (label_num, group) in enumerate(grouped, 1):
        print(f"\nProcessing label group {idx}/{total_groups}: {label_num}")
        
        # Randomly select 5 sentences
        texts = group['text'].tolist()
        sample_size = min(5, len(texts))
        sample_texts = random.sample(texts, sample_size)
        
        # Generate intent label
        intent_label = generate_intent_label(sample_texts)
        
        # Update dataframe
        df.loc[df['predicted_label_num'] == label_num, 'gen_label'] = intent_label
        print(f"Generated label: {intent_label}")
    
    # Save results
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nProcessing complete. Results saved to {output_file}")

if __name__ == "__main__":
    # Get the current directory (where the script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for the input file in the stage3-newdata-assignment directory
    input_file = os.path.join(current_dir, "train_outliners.tsv")
    
    # Create outputs directory in the same location as the script
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"prediction_errors_with_generated_labels_byLDA_{timestamp}.tsv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        print("Please ensure the file 'train_outliners.tsv' is in the same directory as this script")
        print(f"Current directory: {current_dir}")
        exit(1)
        
    process_dataset(input_file, output_file)
import sys
import os
import joblib

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForMaskedLM
from model import CLBert
import logging
from utils.logging_utils import setup_logger
from dataloader import convert_examples_to_features, InputExample, DatasetProcessor, max_seq_lengths
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
from collections import Counter

class IntentClassifier:
    def __init__(self, model_path, bert_path, data_dir, threshold=0.75):
        """Initialize the classifier"""
        self.logger = logging.getLogger('intent_classifier')
        self.threshold = threshold
        self.model_path = model_path
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True, trust_remote_code=True)
        
        # Initialize CLBert model
        self.model = CLBert(bert_path, self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Set maximum sequence length
        self.max_seq_length = max_seq_lengths.get('computerscience', 30)
        
        # Try to load saved state
        if not self.load_state():
            # If no saved state, initialize from scratch
            self.load_pretrained_model(model_path)
            self.load_train_data(data_dir)
            self._train_kmeans()
            # Save complete state
            self.save_state()
        
        self.logger.info(f"Classifier initialized with {len(self.unique_labels)} classes")

    def load_train_data(self, data_dir):
        """Load training data and generate embeddings"""
        self.logger.info("Loading training data...")
        
        # Use DatasetProcessor to load data
        processor = DatasetProcessor()
        train_examples = processor.get_examples_from_dataset(data_dir, 'train')
        self.all_labels = processor.get_labels(data_dir)
        self.unique_labels = np.unique(self.all_labels)
        self.train_texts = [example.text_a for example in train_examples]
        
        # Generate features
        features = convert_examples_to_features(
            train_examples, 
            self.unique_labels, 
            self.max_seq_length, 
            self.tokenizer
        )
        
        # Generate embeddings
        self.model.eval()
        self.embeddings = []
        self.labels = []
        
        batch_size = 32
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            
            # Prepare input data
            inputs = {
                "input_ids": torch.tensor([f.input_ids for f in batch_features], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([f.input_mask for f in batch_features], dtype=torch.long).to(self.device),
                "token_type_ids": torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long).to(self.device)
            }
            
            with torch.no_grad():
                outputs = self.model(inputs)
                batch_embeddings = outputs["features"].cpu().numpy()
                self.embeddings.extend(batch_embeddings)
                self.labels.extend([example.label for example in train_examples[i:i + batch_size]])
        
        self.embeddings = np.array(self.embeddings)
        self.labels = np.array(self.labels)
        
        self.logger.info(f"Generated embeddings shape: {self.embeddings.shape}")

    def load_pretrained_model(self, model_path):
        """Load pretrained model"""
        try:
            # Load model state
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Process key name prefixes
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('bert.'):
                    new_key = key[5:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Load to backbone
            self.model.backbone.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            
            self.logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e

    def get_embedding(self, text):
        """Generate embedding for a single text"""
        example = InputExample(guid="inference-0", text_a=text, text_b=None, label="unknown")
        features = convert_examples_to_features([example], ["unknown"], self.max_seq_length, self.tokenizer)
        
        # Prepare input data
        inputs = {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device),
            "token_type_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
        }
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            features = outputs["features"]
        
        return features.cpu().numpy()

    def classify(self, text):
        """Classify new utterance"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Add input text preprocessing and validation
        if not self._is_valid_input(text):
            return {
                "text": text,
                "predicted_label": "unknown",
                "confidence": 0.0,
                "is_pseudo_label": True,
                "cluster_id": -1,
                "similar_texts": [],
                "error": "Invalid input text"
            }
        
        # Get text embedding
        new_embedding = self.get_embedding(text)
        
        # Calculate similarity with all training samples
        similarities = cosine_similarity(new_embedding, self.embeddings)[0]
        
        # Check if outlier
        if np.max(similarities) < 0.0:  # Increase similarity threshold
            return {
                "text": text,
                "predicted_label": "outlier",
                "confidence": 0.0,
                "is_pseudo_label": True,
                "cluster_id": -1,
                "similar_texts": [],
                "error": "Text identified as outlier"
            }
        
        # Get top K most similar samples
        top_k = 3
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get information about most similar samples
        similar_texts = []
        label_scores = {}
        total_similarity = 0
        
        for idx in top_indices:
            similarity = similarities[idx]
            label = self.labels[idx]
            similar_texts.append({
                "text": self.train_texts[idx],
                "similarity": float(similarity),
                "label": str(label)
            })
            
            # Only consider samples with high similarity for label prediction
            if similarity > 0.5:  # Lower similarity threshold
                label_scores[label] = label_scores.get(label, 0) + similarity
                total_similarity += similarity
        
        # Predict label
        if total_similarity > 0:
            predicted_label = max(label_scores.items(), key=lambda x: x[1])[0]
            confidence = label_scores[predicted_label] / total_similarity
        else:
            # If no sufficiently similar samples, use KMeans clustering result
            cluster_id = self.kmeans.predict(new_embedding)[0]
            cluster_mask = (self.kmeans.labels_ == cluster_id)
            cluster_labels = self.labels[cluster_mask]
            label_counts = Counter(cluster_labels)
            predicted_label = label_counts.most_common(1)[0][0]
            confidence = label_counts[predicted_label] / len(cluster_labels)
        
        # Determine if pseudo label
        is_pseudo = confidence < self.threshold
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "is_pseudo_label": is_pseudo,
            "cluster_id": int(self.kmeans.predict(new_embedding)[0]),
            "similar_texts": similar_texts
        }

    def analyze_clusters(self):
        """Analyze clustering results"""
        clusters_path = 'embeddings/cluster_results.npz'
        if os.path.exists(clusters_path):
            cluster_data = np.load(clusters_path, allow_pickle=True)
            
            # Count cluster sizes
            cluster_sizes = np.bincount(self.kmeans.labels_)
            
            # Get example texts for each cluster
            cluster_examples = {}
            for text, label in zip(self.train_texts, self.kmeans.labels_):
                if label not in cluster_examples:
                    cluster_examples[label] = []
                if len(cluster_examples[label]) < 3:  # Keep 3 examples per cluster
                    cluster_examples[label].append(text)
            
            # Print analysis results
            self.logger.info("\n=== Cluster Analysis Results ===")
            for cluster_id in range(len(self.unique_labels)):
                self.logger.info(f"\nCluster {cluster_id}:")
                self.logger.info(f"Sample count: {cluster_sizes[cluster_id]}")
                self.logger.info("Example texts:")
                for text in cluster_examples.get(cluster_id, []):
                    self.logger.info(f"- {text}")
            
            return {
                'cluster_sizes': cluster_sizes,
                'cluster_examples': cluster_examples,
                'cluster_centers': self.kmeans.cluster_centers_
            }
        else:
            self.logger.warning("Cluster results file not found!")
            return None

    def visualize_clusters(self):
        """Visualize clusters using t-SNE"""
        self.logger.info("Performing t-SNE dimensionality reduction...")
        
        # Use t-SNE for dimensionality reduction to 2D
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Set Chinese font (if needed)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Plot each cluster with different colors
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=self.kmeans.labels_,
            cmap='tab20',
            alpha=0.6
        )
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                            loc="upper right", 
                            title="Clusters")
        plt.gca().add_artist(legend1)
        
        # Plot cluster centers
        centers_2d = tsne.fit_transform(self.kmeans.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0], 
            centers_2d[:, 1], 
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster centers'
        )
        
        plt.title('t-SNE visualization of text embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        # Add statistics
        cluster_sizes = np.bincount(self.kmeans.labels_)
        stats_text = f"Total samples: {len(self.embeddings)}\n"
        stats_text += f"Number of clusters: {len(self.unique_labels)}\n"
        stats_text += f"Largest cluster size: {max(cluster_sizes)}\n"
        stats_text += f"Smallest cluster size: {min(cluster_sizes)}"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Save image
        plt.savefig('embeddings/cluster_visualization.png', dpi=300, bbox_inches='tight')
        self.logger.info("Cluster visualization image saved to embeddings/cluster_visualization.png")
        
        # Save detailed cluster statistics
        cluster_stats = pd.DataFrame({
            'Cluster ID': range(len(self.unique_labels)),
            'Sample count': cluster_sizes,
            'Percentage': cluster_sizes / len(self.embeddings) * 100
        })
        cluster_stats.to_csv('embeddings/cluster_statistics.csv', index=False, encoding='utf-8')
        self.logger.info("Cluster statistics saved to embeddings/cluster_statistics.csv")
        
        return cluster_stats

    def _train_kmeans(self):
        """Train KMeans model"""
        self.logger.info(f"Training new KMeans model, number of clusters: {len(self.unique_labels)}")
        self.kmeans = KMeans(
            n_clusters=len(self.unique_labels), 
            random_state=42,
            n_init=10
        )
        self.kmeans.fit(self.embeddings)
        
        # Save kmeans model
        kmeans_path = 'embeddings/kmeans_model.pkl'
        os.makedirs('embeddings', exist_ok=True)
        joblib.dump(self.kmeans, kmeans_path)
        
        # Save clustering results
        clusters_path = 'embeddings/cluster_results.npz'
        np.savez(
            clusters_path,
            labels=self.kmeans.labels_,
            cluster_centers=self.kmeans.cluster_centers_,
            text_to_cluster=np.array(list(zip(self.train_texts, self.kmeans.labels_)))
        )
        self.logger.info("KMeans model and clustering results saved")

    def _generate_embeddings(self, data_dir):
        """Generate new embeddings"""
        processor = DatasetProcessor()
        train_examples = processor.get_examples_from_dataset(data_dir, 'train')
        self.all_labels = processor.get_labels(data_dir)
        self.unique_labels = np.unique(self.all_labels)
        self.train_texts = [example.text_a for example in train_examples]
        
        self.logger.info("Generating embeddings for training data...")
        self.model.eval()  # Ensure model is in evaluation mode
        
        # Generate embeddings
        self.embeddings = []
        self.labels = []
        
        batch_size = 32
        for i in range(0, len(train_examples), batch_size):
            batch_examples = train_examples[i:i + batch_size]
            features = convert_examples_to_features(
                batch_examples, 
                self.unique_labels, 
                max_seq_length=30, 
                tokenizer=self.tokenizer
            )
            
            # Prepare batch data
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model({
                    "input_ids": input_ids, 
                    "attention_mask": input_mask, 
                    "token_type_ids": segment_ids
                })
                batch_embeddings = outputs["features"].cpu().numpy()
                self.embeddings.extend(batch_embeddings)
                self.labels.extend([example.label for example in batch_examples])
        
        self.embeddings = np.array(self.embeddings)
        self.labels = np.array(self.labels)
        
        self.logger.info(f"Generated embeddings shape: {self.embeddings.shape}")
        
        # Save embeddings, labels, and training texts
        os.makedirs('embeddings', exist_ok=True)
        np.save('embeddings/train_embeddings.npy', self.embeddings)
        np.save('embeddings/train_labels.npy', self.labels)
        with open('embeddings/train_texts.pkl', 'wb') as f:
            pickle.dump(self.train_texts, f)
        self.logger.info("Embeddings-related files saved to embeddings/ directory")

    def save_state(self):
        """Save complete model state"""
        state = {
            'model_state': self.model.state_dict(),
            'embeddings': self.embeddings,
            'train_texts': self.train_texts,
            'labels': self.labels,
            'kmeans': self.kmeans,
            'unique_labels': self.unique_labels,
            'tokenizer_name': self.tokenizer.name_or_path,
            'model_config': {
                'max_seq_length': 30,
                'threshold': self.threshold
            }
        }
        
        # Create save directory
        os.makedirs('embeddings', exist_ok=True)
        
        # Save state
        torch.save(state, 'embeddings/full_state.pt')
        self.logger.info("Complete model state saved")

    def load_state(self):
        """Load complete model state"""
        if os.path.exists('embeddings/full_state.pt'):
            self.logger.info("Found saved model state, loading...")
            try:
                state = torch.load('embeddings/full_state.pt', map_location=self.device)
                
                # Load model state
                self.model.load_state_dict(state['model_state'])
                self.embeddings = state['embeddings']
                self.train_texts = state['train_texts']
                self.labels = state['labels']
                self.kmeans = state['kmeans']
                self.unique_labels = state['unique_labels']
                
                # Verify tokenizer
                if self.tokenizer.name_or_path != state['tokenizer_name']:
                    self.logger.warning("Current tokenizer does not match saved state!")
                
                # Load configuration
                self.threshold = state['model_config']['threshold']
                
                self.logger.info("Successfully loaded model state")
                return True
            except Exception as e:
                self.logger.error(f"Error loading model state: {str(e)}")
                return False
        return False

    def _is_valid_input(self, text):
        """Validate input text"""
        # Check if text contains meaningful words
        words = text.split()
        valid_word_count = sum(1 for word in words if len(word) > 1 and word.isalnum())
        return valid_word_count / len(words) >= 0.5  # At least 50% should be valid words

def main():
    # Set paths
    model_path = r"F:\dev\models\bert-usnid\pytorch_model.bin"
    bert_path = r"F:\dev\models\bert_uncased_L-12_H-768_A-12"
    data_dir = r"D:\dev\DataspellProjects\NID_ACLARR2022\data\computerscience"
    
    # Initialize classifier
    classifier = IntentClassifier(model_path, bert_path, data_dir, threshold=0.9)
    
    # Generate and display cluster visualization
    #cluster_stats = classifier.visualize_clusters()
    #print("\nCluster statistics:")
    #print(cluster_stats.to_string())
    
    # Test some new utterances
    test_texts = [
        #"Identify the H-index of individuals who published a work 2012 or before.",
        #"What are the titles of works referenced by a total of 46 publications?",
        "#！#@@#￥@#%￥#@%#@￥%#@￥%@#￥&……￥……&￥%& ",
        "Can you provide their DBLP key? ",
        "Tom wants to listen to the music",
        "Show the number of publications of Professor Schenkel",
        "What’s the title of work authored by a person whose id is P999 and name is Tom?"
    ]
    
    # Perform classification and print results
    for text in test_texts:
        result = classifier.classify(text)
        print("\n" + "="*50)
        print(f"Input sample: {result['text']}")
        print(f"Predicted label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Is pseudo label: {result['is_pseudo_label']}")
        print(f"Cluster ID: {result['cluster_id']}")
        print("\nMost similar training samples:")
        for item in result['similar_texts']:
            print(f"- Text: {item['text']}")
            print(f"   Similarity: {item['similarity']:.4f}")

if __name__ == "__main__":
    main() 
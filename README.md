# Project Name
HNSCL
## Project Structure
utils/
 file_utils.py # File handling utilities
logging_utils.py # Logging configuration utilities
outputs/ # Generated output files
1 pretrain.py run domain-specific pretraining 
2 train.py run contrastive learning training and k-means clustering
README.md
/user-intent-generation
/embeddings # clustering results on training dataset by loading pre-trained models
inference-cosine.py # load pre-trained models on training dataset and assign new data to one cluster.
generate_labels_LDA.py # generate labels for new clusters by LDA
generate_labels_deepseek.py # generate labels for new clusters by deepseek
generate_labels_llama.py # generate labels for new clusters by llamada
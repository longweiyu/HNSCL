import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import numpy as np
from typing import List, Dict, Any

class FileType(Enum):
    """File type enumeration"""
    LOG = ('log', 'log', 'Experiment log file')
    PLOT_TSNE = ('viz-tsne', 'png', 'T-SNE visualization plot')
    PLOT_TOOL = ('viz-tool', 'png', 'Tool-based visualization plot')
    EVAL = ('eval', 'tsv', 'Evaluation results')
    CLUSTER = ('cluster', 'txt', 'Clustering results')
    CONFIG = ('config', 'json', 'Configuration file')
    RESULTS = ('results', 'csv', 'Experiment results')
    MODEL = ('model', 'pt', 'Model checkpoint')
    PREDICTION = ('predictions', 'tsv', 'Prediction results')

@dataclass
class ExperimentMeta:
    """Experiment metadata"""
    dataset: str
    known_ratio: float
    labeled_ratio: float
    seed: int
    timestamp: str

    @property
    def exp_id(self) -> str:
        """Generate experiment ID"""
        return f"{self.dataset}_k{self.known_ratio}_l{self.labeled_ratio}_s{self.seed}"

def get_run_filename(exp_meta: ExperimentMeta, file_type: FileType, 
                    stage: Optional[str] = None, logger=None,
                    get_name_only: bool = False) -> str:
    """Generate standardized filename
    Args:
        exp_meta: Experiment metadata
        file_type: File type
        stage: Stage identifier
        logger: Logger
        get_name_only: Whether to return filename only without full path
    """
    # Build filename parts
    parts = [
        f"run{exp_meta.timestamp}",   # Run ID
        exp_meta.exp_id,              # Experiment ID
        file_type.value[0],           # File type identifier
    ]
    if stage:
        parts.append(stage)           # Stage identifier (if any)
    
    # Combine filename
    filename = f"{'-'.join(parts)}.{file_type.value[1]}"
    
    # If only need filename, return it
    if get_name_only:
        return filename
        
    # Otherwise return full path
    full_path = os.path.join('outputs', filename)
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Log message
    if logger:
        logger.info(f"Saving {file_type.value[2]} to: {full_path}")
        
    return full_path

class ClusteringUtils:
    @staticmethod
    def save_clustering_visualization(
        exp_meta: ExperimentMeta,
        features: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        logger=None
    ) -> str:
        """Save clustering visualization results
        Args:
            exp_meta: Experiment metadata
            features: Feature matrix
            y_true: True labels
            y_pred: Predicted cluster labels
            logger: Logger
        """
        if logger:
            logger.info("\nGenerating clustering visualization...")
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=exp_meta.seed)
        features_2d = tsne.fit_transform(features)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot true label distribution
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=y_true, cmap='tab20')
        ax1.set_title('True Labels Distribution')
        ax1.set_xlabel('t-SNE dimension 1')
        ax1.set_ylabel('t-SNE dimension 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot predicted cluster distribution
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=y_pred, cmap='tab20')
        ax2.set_title('Predicted Clusters Distribution')
        ax2.set_xlabel('t-SNE dimension 1')
        ax2.set_ylabel('t-SNE dimension 2')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        
        # Save image
        viz_file = get_run_filename(
            exp_meta, 
            FileType.PLOT_TSNE,
            stage='clustering',
            logger=logger
        )
        
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"Saved clustering visualization to: {viz_file}")
        
        return viz_file

    @staticmethod
    def save_clustering_log(
        exp_meta: ExperimentMeta,
        texts: List[str],
        y_pred: np.ndarray,
        num_clusters: int,
        logger=None
    ) -> str:
        """Save clustering log
        Args:
            exp_meta: Experiment metadata
            texts: List of texts
            y_pred: Predicted cluster labels
            num_clusters: Number of clusters
            logger: Logger
        """
        if logger:
            logger.info("\nGenerating clustering log...")
        
        cluster_log_file = get_run_filename(
            exp_meta,
            FileType.CLUSTER,
            stage='clustering',
            logger=logger
        )
        
        # Calculate cluster statistics
        cluster_sizes = Counter(y_pred)
        
        with open(cluster_log_file, 'w', encoding='utf-8') as f:
            f.write("Text Clustering Results:\n")
            f.write("======================\n\n")
            
            f.write("Cluster Statistics:\n")
            for cluster_id in range(num_clusters):
                f.write(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} samples\n")
            f.write("\n======================\n\n")
            
            f.write("Detailed Clustering Results:\n")
            for text, cluster_id in zip(texts, y_pred):
                f.write(f"Cluster {cluster_id}: {text}\n")
        
        if logger:
            logger.info(f"Saved clustering log to: {cluster_log_file}")
            logger.info("\nCluster Statistics:")
            for cluster_id in range(num_clusters):
                logger.info(f"- Cluster {cluster_id}: {cluster_sizes[cluster_id]} samples")
        
        return cluster_log_file

    @staticmethod
    def log_clustering_metrics(results: Dict[str, float], logger=None) -> None:
        """Record clustering evaluation metrics
        Args:
            results: Dictionary of clustering metrics
            logger: Logger
        """
        if logger:
            logger.info('\nClustering Metrics:')
            logger.info(f"- ACC: {results['ACC']:.4f}")
            logger.info(f"- ARI: {results['ARI']:.4f}")
            logger.info(f"- NMI: {results['NMI']:.4f}")

    @staticmethod
    def save_detailed_clustering(
        exp_meta: ExperimentMeta,
        features: np.ndarray,
        y_pred: np.ndarray,
        logger=None,
        title='Clustering Distribution'
    ) -> str:
        """Save clustering visualization results
        Args:
            exp_meta: Experiment metadata
            features: Feature matrix
            y_pred: Predicted cluster labels
            logger: Logger
            title: Chart title
        """
        try:
            if logger:
                logger.info("\nGenerating clustering visualization...")
            
            # Use t-SNE for dimensionality reduction to 2D
            tsne = TSNE(n_components=2, random_state=exp_meta.seed)
            features_2d = tsne.fit_transform(features)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                features_2d[:, 0], 
                features_2d[:, 1], 
                c=y_pred, 
                cmap='tab20', 
                alpha=0.6
            )
            plt.colorbar(scatter)
            
            # Add title and labels
            plt.title(title)
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
            
            # Save image
            viz_file = get_run_filename(
                exp_meta, 
                FileType.PLOT_TOOL,
                stage='clustering',
                logger=logger
            )
            
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if logger:
                logger.info(f"Saved clustering visualization to: {viz_file}")
            
            return viz_file
            
        except Exception as e:
            if logger:
                logger.warning(f"Error generating clustering visualization: {str(e)}")
            raise e
"""
Evaluation module for QA Dataset Clustering.

This module provides tools for evaluating the quality and consistency of clusters
generated by the clustering algorithms. It implements established metrics for
cluster validation in the context of text data clustering.
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.metrics import silhouette_score

from qadst.logging import get_logger

logger = get_logger(__name__)


class ClusterEvaluator:
    """
    Evaluates the quality of text clusters using established metrics.

    This class provides methods to assess cluster quality using metrics like
    silhouette score, which measures how similar an object is to its own cluster
    compared to other clusters.
    """

    def __init__(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        cluster_assignments: List[int],
        model_name: str,
    ):
        """
        Initialize the cluster evaluator.

        Args:
            texts: List of text strings that were clustered
            embeddings: Numpy array of embeddings for each text
            cluster_assignments: List of cluster IDs for each text
            model_name: Name of the clustering model (e.g., "DP", "PYP")
        """
        self.texts = texts
        self.embeddings = embeddings
        self.cluster_assignments = cluster_assignments
        self.model_name = model_name
        self.unique_clusters = list(set(cluster_assignments))

        # Validate inputs
        if len(texts) != len(embeddings) or len(texts) != len(cluster_assignments):
            raise ValueError(
                "Length mismatch: texts, embeddings, and cluster_assignments "
                "must have the same length"
            )

        logger.info(
            f"Initialized cluster evaluator for {model_name} with {len(texts)} texts "
            f"and {len(self.unique_clusters)} clusters"
        )

    def calculate_silhouette_score(self) -> Union[float, int]:
        """
        Calculate the silhouette score for the clustering.

        The silhouette score measures how similar an object is to its own cluster
        compared to other clusters. The score ranges from -1 to 1, where:
        - A high value (close to 1) indicates the object is well matched to its cluster
        - A value near 0 indicates the object is on or very close to the decision
          boundary
        - A negative value indicates the object might be assigned to the wrong cluster

        Returns:
            Silhouette score as a float
        """
        # We need at least 2 clusters and each cluster must have at least 2 samples
        if len(self.unique_clusters) < 2:
            logger.warning(
                f"Cannot calculate silhouette score: only "
                f"{len(self.unique_clusters)} cluster found"
            )
            return 0.0

        # Count samples per cluster
        cluster_counts = {}
        for cluster_id in self.cluster_assignments:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        # Check if any cluster has only one sample
        single_sample_clusters = [c for c, count in cluster_counts.items() if count < 2]
        if single_sample_clusters:
            logger.warning(
                f"Cannot calculate silhouette score: "
                f"{len(single_sample_clusters)} clusters have fewer than 2 samples"
            )
            return 0.0

        try:
            score = silhouette_score(
                self.embeddings, self.cluster_assignments, metric="cosine"
            )
            logger.info(f"Silhouette score for {self.model_name}: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            return 0.0

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Returns:
            Dictionary containing evaluation metrics
        """
        report = {
            "basic_metrics": {
                "model_name": self.model_name,
                "num_texts": len(self.texts),
                "num_clusters": len(self.unique_clusters),
            },
            "silhouette_score": self.calculate_silhouette_score(),
        }

        return report


def save_evaluation_report(
    report: Dict[str, Any], output_dir: str, filename: str = "evaluation_report.json"
) -> str:
    """
    Save the evaluation report to a JSON file.

    Args:
        report: Dictionary containing the evaluation report
        output_dir: Directory to save the report
        filename: Name of the output file

    Returns:
        Path to the saved report file
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_path}")
    return output_path

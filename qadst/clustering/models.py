"""
Clustering models for text data using Dirichlet Process and Pitman-Yor Process.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from qadst.clustering.cache import EmbeddingCache
from qadst.logging import get_logger

logger = get_logger(__name__)


class DirichletProcess:
    """Dirichlet Process clustering implementation."""

    def __init__(
        self,
        alpha: float,
        base_measure: Optional[Tensor] = None,
        similarity_metric: Optional[Callable[[str, Tensor], float]] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        self.alpha = alpha
        self.base_measure = base_measure
        self.clusters: list[int] = []
        self.cluster_params: list[Tensor] = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_metric = (
            similarity_metric if similarity_metric else self.bert_similarity
        )

        self.cache = cache
        if self.cache:
            self.cache.load_cache()

    def get_embedding(self, text: str) -> Tensor:
        # Try to get from cache first
        if self.cache and text in self.cache:
            embedding = self.cache.get(text)
            if embedding is not None:
                return embedding

        # Generate new embedding
        embedding = self.model.encode(text)

        # Store in cache if provider available
        if self.cache:
            self.cache.set(text, embedding)

        return embedding

    def save_embedding_cache(self):
        if self.cache:
            self.cache.save_cache()

    def bert_similarity(self, text, cluster_param):
        text_embedding = self.get_embedding(text)
        cluster_embedding = cluster_param
        similarity = 1 - cosine(text_embedding, cluster_embedding)
        return max(0.0, similarity)

    def sample_new_cluster(self, text):
        return self.get_embedding(text)

    def assign_cluster(self, text):
        probs = []
        total_points = len(self.clusters)

        for i, params in enumerate(self.cluster_params):
            cluster_size = self.clusters.count(i)
            prob = (
                cluster_size / (self.alpha + total_points)
            ) * self.similarity_metric(text, params)
            probs.append(max(0.0, prob))

        new_cluster_prob = (self.alpha / (self.alpha + total_points)) * 1.0
        probs.append(new_cluster_prob)

        probs = np.array(probs)
        if probs.sum() <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)
        if choice == len(self.cluster_params):
            new_params = self.sample_new_cluster(text)
            self.cluster_params.append(new_params)
            self.clusters.append(len(self.cluster_params) - 1)
        else:
            self.clusters.append(choice)

    def fit(self, texts: List[str]) -> Tuple[List[int], List[Tensor]]:
        """
        Train the Dirichlet Process model on the given text data.

        Args:
            texts: List of text strings to cluster

        Returns:
            Tuple containing (cluster_assignments, cluster_parameters)
        """
        logger.info(f"Processing {len(texts)} texts...")
        for text in tqdm(texts, desc="Clustering"):
            self.assign_cluster(text)

        self.save_embedding_cache()

        return self.clusters, self.cluster_params


class PitmanYorProcess(DirichletProcess):
    """Pitman-Yor Process clustering implementation."""

    def __init__(
        self,
        alpha: float,
        sigma: float,
        base_measure: Optional[Tensor] = None,
        similarity_metric: Optional[Callable[[str, Tensor], float]] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        super().__init__(alpha, base_measure, similarity_metric, cache)
        self.sigma = sigma
        # Keep track of cluster sizes for faster access
        self.cluster_sizes = {}

    def assign_cluster(self, text):
        """Uses Pitman-Yor process probability calculations."""
        probs = []
        total_points = len(self.clusters)

        # Pre-compute the embedding once
        text_embedding = self.get_embedding(text)

        # Update cluster sizes dictionary
        if not hasattr(self, "cluster_sizes") or self.cluster_sizes is None:
            self.cluster_sizes = {}
            for i in range(len(self.cluster_params)):
                self.cluster_sizes[i] = self.clusters.count(i)

        for i, params in enumerate(self.cluster_params):
            # Use the cached cluster size instead of counting each time
            cluster_size = self.cluster_sizes.get(i, 0)
            adjusted_size = max(self.sigma, cluster_size)

            # Calculate similarity directly with embeddings for speed
            similarity = 1 - cosine(text_embedding, params)
            similarity = max(0.0, similarity)

            prob = (
                (adjusted_size - self.sigma) / (self.alpha + total_points) * similarity
            )
            probs.append(max(0.0, prob))

        new_cluster_prob = (
            (self.alpha + self.sigma * len(self.cluster_params))
            / (self.alpha + total_points)
        ) * 1.0
        probs.append(new_cluster_prob)

        probs = np.array(probs)
        if probs.sum() <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)
        if choice == len(self.cluster_params):
            # Use the already computed embedding
            self.cluster_params.append(text_embedding)
            self.clusters.append(len(self.cluster_params) - 1)
            # Update cluster sizes
            self.cluster_sizes[len(self.cluster_params) - 1] = 1
        else:
            self.clusters.append(choice)
            # Update cluster sizes
            self.cluster_sizes[choice] = self.cluster_sizes.get(choice, 0) + 1

    def fit(self, texts: List[str]) -> Tuple[List[int], List[Tensor]]:
        """
        Optimized version of fit for PitmanYorProcess.

        Args:
            texts: List of text strings to cluster

        Returns:
            Tuple containing (cluster_assignments, cluster_parameters)
        """
        logger.info(f"Processing {len(texts)} texts with optimized PitmanYorProcess...")

        # Initialize cluster sizes dictionary
        self.cluster_sizes = {}

        # Process texts in batches for better progress reporting
        batch_size = 100
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            for text in tqdm(
                batch, desc=f"Clustering batch {batch_num}/{total_batches}"
            ):
                self.assign_cluster(text)

        self.save_embedding_cache()
        return self.clusters, self.cluster_params

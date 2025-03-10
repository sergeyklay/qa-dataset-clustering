"""Base functionality for QA dataset clustering."""

import csv
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_openai import ChatOpenAI

from .embeddings import get_embeddings_model
from .filters import ProductDevelopmentFilter

logger = logging.getLogger(__name__)


class BaseClusterer(ABC):
    """Abstract base class for question-answer clustering algorithms.

    This class provides common functionality for clustering question-answer pairs,
    including deduplication, filtering, and processing datasets. Subclasses must
    implement the clustering algorithm-specific methods.

    Attributes:
        output_dir: Directory to save output files
        embedding_model_name: Name of the embedding model
        embeddings_model: Initialized embedding model instance
        filter_enabled: Whether filtering is enabled
        filter_cache: Cache for filter results to avoid redundant LLM calls
        llm: Language model for filtering and topic labeling
    """

    def __init__(
        self,
        embedding_model_name: str,
        llm_model_name: Optional[str] = None,
        output_dir: str = "./output",
        filter_enabled: bool = True,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: Optional[float] = None,
        keep_noise: bool = False,
        cluster_selection_method: str = "eom",
    ):
        """Initialize the clusterer.

        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Optional name of the LLM to use for filtering and labeling
            output_dir: Directory to save output files
            filter_enabled: Whether to filter out engineering-focused questions
            min_cluster_size: Minimum size of clusters (if None, auto-calculated)
            min_samples: HDBSCAN min_samples parameter (default: 5)
            cluster_selection_epsilon: HDBSCAN epsilon parameter (default: 0.3)
            keep_noise: Whether to keep noise points unclustered (default: False)
            cluster_selection_method: HDBSCAN cluster selection method (default: "eom")
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.output_dir = output_dir
        self.filter_enabled = filter_enabled
        self.embedding_cache = {}
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.keep_noise = keep_noise
        self.cluster_selection_method = cluster_selection_method

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize embedding model using the factory function
        self.embeddings_model = get_embeddings_model(embedding_model_name)

        # Initialize LLM if provided
        self.llm = None
        if llm_model_name:
            self.llm = ChatOpenAI(
                model=llm_model_name,
                temperature=0,
            )

        # Initialize filter
        self.product_dev_filter = ProductDevelopmentFilter(llm=self.llm)

    def load_qa_pairs(self, csv_path: str) -> List[Tuple[str, str]]:
        """Load question-answer pairs from a CSV file.

        Args:
            csv_path: Path to the CSV file containing question-answer pairs

        Returns:
            List of (question, answer) tuples

        Example:
            Given a CSV file 'questions.csv' with content:
            ```
            question,answer
            "How do I reset my password?","Click the 'Forgot Password' link."
            "What payment methods do you accept?","We accept credit cards and PayPal."
            ```

            >>> clusterer = HDBSCANQAClusterer("text-embedding-3-large")
            >>> qa_pairs = clusterer.load_qa_pairs("questions.csv")
            >>> print(qa_pairs[0])
            ('How do I reset my password?', 'Click the 'Forgot Password' link.')
        """
        qa_pairs = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            if (
                len(header) < 2
                or "question" not in header[0].lower()
                or "answer" not in header[1].lower()
            ):
                raise ValueError(
                    "CSV file must have 'question' and 'answer' columns. "
                    f"Found: {header}"
                )

            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    qa_pairs.append((row[0], row[1]))

        return qa_pairs

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        The cosine similarity measures the cosine of the angle between two vectors,
        providing a similarity score between -1 and 1, where 1 means identical,
        0 means orthogonal, and -1 means opposite.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity value between -1 and 1

        Example:
            >>> clusterer = BaseClusterer(embedding_model_name="text-embedding-3-large")
            >>> vec1 = np.array([0.1, 0.2, 0.3])
            >>> vec2 = np.array([0.2, 0.3, 0.5])
            >>> similarity = clusterer.calculate_cosine_similarity(vec1, vec2)
            >>> print(f"Similarity: {similarity:.4f}")
            Similarity: 0.9922
        """
        # Convert to numpy arrays if they aren't already
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.dot(vec1_np, vec2_np) / (
            np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
        )

    def _calculate_deterministic_hash(self, items: List[str]) -> str:
        """Calculate a stable hash for an items list."""
        # Sort to make order-independent (if needed)
        sorted_items = sorted(items)
        combined = "".join(sorted_items).encode("utf-8")

        # SHA-256 produces 64-character hex digest
        return hashlib.sha256(combined).hexdigest()

    def deduplicate_questions(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Remove semantically duplicate questions using embedding similarity.

        This method identifies and removes semantically similar questions by:
        1. Computing embeddings for all questions
        2. Calculating pairwise cosine similarities
        3. Grouping questions that exceed the similarity threshold
        4. Keeping one representative from each group

        Uses embedding caching to avoid recomputing embeddings across runs.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            List of deduplicated (question, answer) tuples

        Example:
            >>> clusterer = HDBSCANQAClusterer("text-embedding-3-large")
            >>> qa_pairs = [
            ...     ("How do I reset my password?", "Click 'Forgot Password'"),
            ...     ("How can I change my password?", "Use the 'Forgot Password' link"),
            ...     ("What payment methods do you accept?", "We accept credit cards")
            ... ]
            >>> deduplicated = clusterer.deduplicate_questions(qa_pairs)
            >>> # Only two questions remain, as the first two are semantically similar
            >>> len(deduplicated)
            2
        """
        import time

        if not qa_pairs:
            return []

        start = time.time()
        questions = [q for q, _ in qa_pairs]

        # Use cached embeddings if available
        questions_hash = self._calculate_deterministic_hash(questions)
        cache_key = f"dedup_{self.embedding_model_name}_{questions_hash}"
        question_embeddings = self.get_embeddings(questions, cache_key)

        similarity_threshold = 0.85
        duplicate_groups = {}
        duplicate_indices = set()
        total_questions = len(questions)
        processed_count = 0
        duplicate_count = 0

        logger.info("Starting duplicate detection")
        for i in range(len(questions)):
            processed_count += 1
            if i in duplicate_indices:
                continue

            duplicate_groups[i] = [i]

            for j in range(i + 1, len(questions)):
                if j in duplicate_indices:
                    continue

                similarity = self.calculate_cosine_similarity(
                    question_embeddings[i], question_embeddings[j]
                )

                if similarity > similarity_threshold:
                    duplicate_indices.add(j)
                    duplicate_groups[i].append(j)
                    duplicate_count += 1
                    logger.debug(
                        f"[{processed_count}/{total_questions}] Found duplicate: "
                        f"'{questions[j]}' similar to '{questions[i]}'"
                    )

        deduplicated_pairs = []

        for canonical_idx, group in duplicate_groups.items():
            deduplicated_pairs.append(qa_pairs[canonical_idx])

        logger.info(
            f"Found {duplicate_count} duplicates out of {total_questions} questions"
        )

        logger.info(f"Deduplication time: {time.time()-start:.2f}s")
        return deduplicated_pairs

    def filter_questions(
        self,
        qa_pairs: List[Tuple[str, str]],
        batch_size: int = 20,
        use_llm: bool = True,
        cache_file: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Filter out questions for product development teams.

        Uses an LLM to classify questions as either product development-focused or
        client-focused. Processes questions in batches for efficiency and maintains
        a cache to avoid redundant LLM calls.

        This filter distinguishes between:
        1. Client engineering questions (which are kept)
        2. Product development team questions (which are filtered out)

        Args:
            qa_pairs: List of (question, answer) tuples
            batch_size: Number of questions to process in each batch
            use_llm: Whether to use an LLM for filtering
            cache_file: Optional path to a cache file to persist filter results

        Returns:
            List of filtered (question, answer) tuples for clients

        Example:
            >>> clusterer = HDBSCANQAClusterer(
            ...     embedding_model_name="text-embedding-3-large",
            ...     llm_model_name="gpt-4o"
            ... )
            >>> qa_pairs = [
            ...     ("How do I reset my password?", "Click 'Forgot Password'"),
            ...     ("What's the expected API latency in EU region?", "Under 100ms"),
            ...     ("How do I contact support?", "Email support@example.com")
            ... ]
            >>> filtered = clusterer.filter_questions(qa_pairs)
            >>> # The second question is filtered out as engineering-focused
            >>> len(filtered)
            2
            >>> filtered[0][0]
            'How do I reset my password?'
        """
        if not use_llm or not self.llm:
            logger.warning("LLM not provided or filtering disabled, skipping filter")
            return qa_pairs

        # Use the ProductDevelopmentFilter to process questions
        kept_pairs, filtered_pairs = self.product_dev_filter.process_questions(
            qa_pairs, batch_size, cache_file
        )

        # Save engineering questions to a separate file
        engineering_file = os.path.join(self.output_dir, "engineering_questions.csv")
        with open(engineering_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for q, a in filtered_pairs:
                writer.writerow([q, a])
        logger.debug(f"Saved filtered questions to {engineering_file}")

        return kept_pairs

    @abstractmethod
    def cluster_questions(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Cluster questions based on semantic similarity.

        This abstract method must be implemented by subclasses to provide
        algorithm-specific clustering functionality.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results in the requested format
        """
        pass

    @abstractmethod
    def cluster_method(self) -> str:
        """Return the name of the clustering method.

        This abstract method must be implemented by subclasses to provide
        the name of the clustering algorithm used.

        Returns:
            String name of the clustering method
        """
        pass

    def process_dataset(self, csv_path: str) -> Dict[str, Any]:
        """Process a CSV file containing QA pairs through the full pipeline.

        This method orchestrates the complete processing pipeline:
        1. Load QA pairs from CSV
        2. Deduplicate questions
        3. Filter out engineering questions (if enabled)
        4. Cluster the questions
        5. Save results to JSON and CSV files

        Args:
            csv_path: Path to the CSV file containing question-answer pairs

        Returns:
            Dict containing clustering results and paths to output files

        Example:
            >>> clusterer = HDBSCANQAClusterer(
            ...     embedding_model_name="text-embedding-3-large",
            ...     llm_model_name="gpt-4o",
            ...     output_dir="./results"
            ... )
            >>> results = clusterer.process_dataset("questions.csv")
            >>> # Results contain clustering results and output file paths
            >>> print(results.keys())
            dict_keys(['clustering_results', 'json_output_path', 'csv_output_path',
            ... 'deduplicated_count', 'original_count', 'filtered_count',
            ... 'filter_cache_path', 'num_clusters'])
            >>> # Check the clustering results
            >>> print(results['clustering_results'].keys())
            dict_keys(['clusters'])
            >>> # Check the number of clusters found
            >>> print(results['num_clusters'])
            5
            >>> # Check the output file paths
            >>> print(f"JSON output: {results['json_output_path']}")
            JSON output: ./results/qa_clusters.json
            >>> print(f"CSV output: {results['csv_output_path']}")
            CSV output: ./results/qa_cleaned.csv
        """
        start = time.time()

        logger.info(f"Loading QA pairs from {csv_path}")
        qa_pairs = self.load_qa_pairs(csv_path)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs in {time.time()-start:.2f}s")

        logger.info("Deduplicating questions")
        deduplicated_pairs = self.deduplicate_questions(qa_pairs)
        logger.info(f"Deduplicated to {len(deduplicated_pairs)} QA pairs")

        filtered_pairs = deduplicated_pairs

        if self.filter_enabled:
            logger.info("Filtering out engineering-focused questions")
            cache_file = os.path.join(self.output_dir, "filter_cache.json")
            filter_start = time.time()
            filtered_pairs = self.filter_questions(
                deduplicated_pairs,
                batch_size=20,
                use_llm=self.llm is not None,
                cache_file=cache_file,
            )
            logger.info(f"Filtering time: {time.time()-filter_start:.2f}s")
            logger.info(f"Retained {len(filtered_pairs)} client-focused QA pairs")
        else:
            logger.info("Filtering is disabled, skipping")

        logger.info(f"Clustering questions using {self.cluster_method()}")
        clustering_results = self.cluster_questions(filtered_pairs)

        # Get the number of clusters
        num_clusters = len(clustering_results.get("clusters", []))
        logger.debug(f"Found {num_clusters} clusters")

        json_output_path = os.path.join(self.output_dir, "qa_clusters.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(clustering_results, f, indent=2)
        logger.debug(f"Saved clustering results to {json_output_path}")

        csv_output_path = os.path.join(self.output_dir, "qa_cleaned.csv")
        with open(csv_output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for question, answer in filtered_pairs:
                writer.writerow([question, answer])
        logger.debug(f"Saved cleaned QA pairs to {csv_output_path}")

        result = {
            "clustering_results": clustering_results,
            "json_output_path": json_output_path,
            "csv_output_path": csv_output_path,
            "deduplicated_count": len(deduplicated_pairs),
            "original_count": len(qa_pairs),
            "num_clusters": num_clusters,
        }

        if self.filter_enabled:
            result["filtered_count"] = len(filtered_pairs)
            result["filter_cache_path"] = cache_file

        logger.info(f"Processing time: {time.time()-start:.2f}s")
        return result

    # Move this to embeddings.py
    def get_embeddings(
        self, questions: List[str], cache_key: Optional[str] = None
    ) -> List[np.ndarray]:
        """Get embeddings for a list of questions, using cache when available.

        This method checks if embeddings are already cached (either in memory or
        on disk) before computing them, which can significantly improve performance
        across runs.

        Args:
            questions: List of questions to embed
            cache_key: Optional key to use for caching (e.g., a hash of the dataset)
                       If None, caching will not be used.

        Returns:
            List of numpy arrays containing the embeddings
        """
        if not questions:
            return []

        def embed_questions(items: List[str]) -> List[np.ndarray]:
            """Embed questions using the embedding model."""
            embeddings_list = self.embeddings_model.embed_documents(items)
            embeddings = [np.array(emb) for emb in embeddings_list]
            return embeddings

        if cache_key is None:
            return embed_questions(questions)

        # Check in-memory cache first
        if cache_key in self.embedding_cache:
            logger.info(f"Using in-memory cache for embeddings (key: {cache_key})")
            return self.embedding_cache[cache_key]

        # Check disk cache
        cache_dir = os.path.join(self.output_dir, "embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_file):
            try:
                logger.debug(f"Loading embeddings from cache file: {cache_file}")
                embeddings_array = np.load(cache_file, allow_pickle=True)
                embeddings = [np.array(emb) for emb in embeddings_array]

                # Store in memory cache for faster access next time
                self.embedding_cache[cache_key] = embeddings
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings from cache: {e}")

        # Compute embeddings if not in cache
        logger.info(f"Computing embeddings for {len(questions)} questions")
        embeddings = embed_questions(questions)

        # Save to disk cache
        try:
            logger.debug(f"Saving embeddings to cache file: {cache_file}")
            np.save(cache_file, embeddings, allow_pickle=True)

            # Store in memory cache
            self.embedding_cache[cache_key] = embeddings
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {e}")

        return embeddings

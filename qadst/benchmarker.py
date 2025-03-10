"""Benchmarking and evaluation tools for QA dataset clustering.

This module provides functionality for evaluating the quality of clustering results,
generating topic labels for clusters, and producing comprehensive reports on cluster
quality metrics such as coherence, separation, and density.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .embeddings import get_embeddings_model
from .reporters import (
    ConsoleReporter,
    CSVReporter,
    ReportData,
    ReporterRegistry,
)

logger = logging.getLogger(__name__)


class ClusterBenchmarker:
    """Evaluates and analyzes the quality of clustering results.

    This class provides methods to assess clustering quality using various metrics,
    generate topic labels for clusters, and create comprehensive reports. It supports
    both statistical evaluation metrics and semantic coherence measures.

    The benchmarker can use:
    1. Standard clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
    2. Semantic coherence based on embedding similarity
    3. LLM-based topic labeling for interpretability
    4. TF-IDF/NMF-based topic extraction as a fallback

    Attributes:
        embeddings_model: Model for generating embeddings
        llm: Language model for topic labeling
        output_dir: Directory to save output files
        reporter_registry: Registry of reporters for output
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        output_dir: str = "./output",
    ):
        """Initialize the benchmarker.

        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the LLM model to use for topic labeling
            output_dir: Directory to save output files
        """
        self.embedding_model_name = embedding_model_name
        self.embeddings_model = None
        if embedding_model_name:
            try:
                self.embeddings_model = get_embeddings_model(embedding_model_name)
                logger.info(f"Initialized embeddings model: {embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings model: {e}")

        self.llm = None
        if llm_model_name:
            try:
                self.llm = ChatOpenAI(model=llm_model_name, temperature=0.0)
                logger.info(f"Initialized LLM with model: {llm_model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize reporter registry with default reporters
        self.reporter_registry = ReporterRegistry()
        self.reporter_registry.register("csv", CSVReporter(output_dir), enabled=True)
        self.reporter_registry.register(
            "console", ConsoleReporter(output_dir), enabled=True
        )

    def load_clusters(self, json_path: str) -> Dict[str, Any]:
        """Load clusters from a JSON file.

        Args:
            json_path: Path to the JSON file containing clustering results

        Returns:
            Dict containing clustering results
        """
        with open(json_path, "r") as f:
            return json.load(f)

    def load_qa_pairs(self, csv_path: str) -> List[Tuple[str, str]]:
        """Load question-answer pairs from a CSV file.

        Args:
            csv_path: Path to the CSV file containing question-answer pairs

        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    qa_pairs.append((row[0], row[1]))
        return qa_pairs

    def extract_embeddings_from_qa_pairs(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """Extract embeddings from QA pairs using the embeddings model.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Array of embeddings for the questions
        """
        # Initialize embeddings model if not already initialized
        if self.embeddings_model is None:
            if self.embedding_model_name:
                try:
                    self.embeddings_model = get_embeddings_model(
                        self.embedding_model_name
                    )
                    logger.info(
                        f"Initialized embeddings model: {self.embedding_model_name}"
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize embeddings model: {e}")
                    raise ValueError(f"Failed to initialize embeddings model: {e}")
            else:
                raise ValueError("Embedding model name not provided")

        questions = [q for q, _ in qa_pairs]
        return np.array(self.embeddings_model.embed_documents(questions))

    def prepare_cluster_data(
        self, clusters: Dict[str, Any], embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for cluster quality evaluation.

        Maps cluster assignments to the original questions and creates arrays
        of embeddings and corresponding cluster labels for metric calculation.

        Args:
            clusters: Dict containing clustering results
            embeddings: Array of embeddings

        Returns:
            Tuple of (embeddings_array, labels_array)
        """
        question_to_idx = {}
        for i, (q, _) in enumerate(self.qa_pairs):
            question_to_idx[q] = i

        labels = np.full(len(self.qa_pairs), -1)  # Default to noise

        for cluster_idx, cluster in enumerate(clusters["clusters"]):
            for qa_pair in cluster["source"]:
                question = qa_pair["question"]
                if question in question_to_idx:
                    labels[question_to_idx[question]] = cluster_idx

        return embeddings, labels

    def calculate_metrics(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard cluster quality metrics.

        Computes three widely used clustering evaluation metrics:
        1. Davies-Bouldin Index: Lower values indicate better clustering
        2. Calinski-Harabasz Index: Higher values indicate better clustering
        3. Silhouette Score: Higher values (-1 to 1) indicate better clustering

        Also calculates the noise ratio (proportion of points not assigned to clusters).

        Args:
            embeddings: Array of embeddings
            labels: Array of cluster labels

        Returns:
            Dict containing metrics

        Example:
            >>> import numpy as np
            >>> benchmarker = ClusterBenchmarker()
            >>> # Create sample embeddings (20 samples, 5 dimensions)
            >>> embeddings = np.random.rand(20, 5)
            >>> # Create sample cluster labels (3 clusters + noise points)
            >>> labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, -1, -1,
            ...                    0, 1, 2, 0, 1, 2, -1, 0])
            >>> metrics = benchmarker.calculate_metrics(embeddings, labels)
            >>> # Print the metrics
            >>> for metric, value in metrics.items():
            ...     print(f"{metric}: {value:.4f}")
            noise_ratio: 0.2000
            davies_bouldin_score: 1.2345
            calinski_harabasz_score: 2.3456
            silhouette_score: 0.3456
        """
        non_noise_mask = labels != -1
        non_noise_embeddings = embeddings[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]

        metrics = {}
        metrics["noise_ratio"] = 1.0 - (np.sum(non_noise_mask) / len(labels))

        if len(np.unique(non_noise_labels)) <= 1 or len(non_noise_embeddings) == 0:
            metrics["davies_bouldin_score"] = float("nan")
            metrics["calinski_harabasz_score"] = float("nan")
            metrics["silhouette_score"] = float("nan")
            return metrics

        metrics["davies_bouldin_score"] = davies_bouldin_score(
            non_noise_embeddings, non_noise_labels
        )

        metrics["calinski_harabasz_score"] = calinski_harabasz_score(
            non_noise_embeddings, non_noise_labels
        )

        metrics["silhouette_score"] = silhouette_score(
            non_noise_embeddings, non_noise_labels
        )

        return metrics

    def calculate_cluster_coherence(self, cluster_questions: List[str]) -> float:
        """Calculate semantic coherence score for a cluster.

        Measures how semantically similar the questions within a cluster are to each
        other by computing the average pairwise cosine similarity between their
        embeddings. Higher values indicate more coherent clusters.

        Args:
            cluster_questions: List of questions in the cluster

        Returns:
            Coherence score (average pairwise similarity) between 0 and 1

        Example:
            >>> benchmarker = ClusterBenchmarker("text-embedding-3-large")
            >>> # Questions about password management (semantically similar)
            >>> coherent_cluster = [
            ...     "How do I reset my password?",
            ...     "What's the process for changing my password?",
            ...     "I forgot my password, how can I recover my account?"
            ... ]
            >>> coherence = benchmarker.calculate_cluster_coherence(coherent_cluster)
            >>> print(f"Coherence score: {coherence:.4f}")
            Coherence score: 0.8765  # High score indicates coherent cluster

            >>> # Mixed questions (semantically diverse)
            >>> mixed_cluster = [
            ...     "How do I reset my password?",
            ...     "What payment methods do you accept?",
            ...     "How do I cancel my subscription?"
            ... ]
            >>> coherence = benchmarker.calculate_cluster_coherence(mixed_cluster)
            >>> print(f"Coherence score: {coherence:.4f}")
            Coherence score: 0.5432  # Lower score for less coherent cluster
        """
        if self.embeddings_model is None:
            raise ValueError("Embeddings model not provided")

        if len(cluster_questions) <= 1:
            return 1.0  # Perfect coherence for single-item clusters

        embeddings = np.array(self.embeddings_model.embed_documents(cluster_questions))

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)

        return float(np.mean(similarities))

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        The cosine similarity measures the cosine of the angle between two vectors,
        providing a similarity score between -1 and 1, where 1 means identical,
        0 means orthogonal, and -1 means opposite.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity value between -1 and 1
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _generate_llm_topic_label(
        self, questions: List[str], previous_topics: Optional[List[str]] = None
    ) -> str:
        """Generate a descriptive topic label using an LLM.

        Uses a language model to analyze a set of questions and generate a concise,
        specific topic label that captures the common theme. The prompt is designed
        to produce distinctive, non-generic labels.

        Args:
            questions: List of questions to generate a topic label for
            previous_topics: List of previously generated topic labels to avoid overlap

        Returns:
            Topic label as a string
        """
        if not self.llm:
            return "No LLM Available"

        formatted_questions = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(questions)]
        )

        previous_topics_text = ""
        if previous_topics and len(previous_topics) > 0:
            previous_topics_text = (
                "\n\nPreviously generated topics (AVOID SEMANTIC OVERLAP WITH THESE):\n"
            )
            topics_formatted = [f"- {topic}" for topic in previous_topics]
            previous_topics_text += "\n".join(topics_formatted)

        prompt_template = PromptTemplate(
            input_variables=["questions", "previous_topics"],
            template="""
            You are an expert taxonomist specializing in creating precise, distinctive category labels.

            Below is a list of questions that belong to the same cluster:

            {questions}

            Your task is to create a HIGHLY SPECIFIC topic label (2-4 words) that:
            1. Precisely captures what makes these questions UNIQUE compared to other topics
            2. Uses concrete, specific terminology rather than generic terms
            3. Avoids using the product name in the label
            4. Focuses on the distinctive FUNCTION or CONCEPT these questions address
            5. Is concise and memorable
            6. Is SEMANTICALLY DISTINCT from any previously generated topics

            BAD EXAMPLES (too generic):
            - "Document Management"
            - "User Features"
            - "Account Settings"
            - "Acme Features"

            GOOD EXAMPLES (specific and distinctive):
            - "Offline Deployment Options"
            - "Team Permission Hierarchy"
            - "Template Reusability"
            - "Audit Trail Functionality"
            - "CRM Integration Methods"
            {previous_topics}

            Respond ONLY with the final topic label, nothing else.
            """,  # noqa: E501
        )

        chain = prompt_template | self.llm

        try:
            response = chain.invoke(
                {
                    "questions": formatted_questions,
                    "previous_topics": previous_topics_text,
                }
            )

            if hasattr(response, "content"):
                topic_label = str(response.content).strip().strip('"').strip("'")
            else:
                topic_label = str(response).strip().strip('"').strip("'")

            if len(topic_label) > 50:
                topic_label = topic_label[:47] + "..."

            return topic_label
        except Exception as e:
            logger.warning(f"Error generating topic label: {e}")
            return "LLM Error"
        else:
            return "No LLM Available"

    def _collect_cluster_questions(
        self, clusters: Dict[str, Any], max_questions_per_cluster: int
    ) -> Dict[int, List[str]]:
        """Collect questions from each cluster for topic labeling.

        Args:
            clusters: Dict containing clustering results
            max_questions_per_cluster: Maximum number of questions to use for labeling

        Returns:
            Dict mapping cluster IDs to lists of questions
        """
        all_cluster_questions = {}
        for cluster in clusters["clusters"]:
            cluster_id = cluster["id"]
            questions = [qa["question"] for qa in cluster["source"]]
            if questions:
                all_cluster_questions[cluster_id] = questions[
                    :max_questions_per_cluster
                ]
        return all_cluster_questions

    def _generate_tfidf_topic_label(
        self, questions: List[str], n_topics: int, n_top_words: int
    ) -> str:
        """Generate a topic label using TF-IDF/NMF method.

        Args:
            questions: List of questions in the cluster
            n_topics: Number of topics to extract
            n_top_words: Number of top words to include in the label

        Returns:
            Topic label string
        """
        # Extract important words using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(questions)
        feature_names = vectorizer.get_feature_names_out()

        # For clusters with enough documents, use NMF to extract topics
        if len(questions) >= 3:
            return self._extract_nmf_topics(
                tfidf_matrix, feature_names, n_topics, n_top_words
            )
        else:
            # For small clusters, use the top TF-IDF terms directly
            return self._extract_tfidf_topics(tfidf_matrix, feature_names, n_top_words)

    def _extract_nmf_topics(
        self, tfidf_matrix, feature_names, n_topics: int, n_top_words: int
    ) -> str:
        """Extract topics using Non-negative Matrix Factorization.

        Args:
            tfidf_matrix: TF-IDF matrix of documents
            feature_names: Feature names from vectorizer
            n_topics: Number of topics to extract
            n_top_words: Number of top words to include

        Returns:
            Topic label string
        """
        # Non-negative Matrix Factorization for topic modeling
        nmf_model = NMF(
            n_components=min(n_topics, tfidf_matrix.shape[0]), random_state=42
        )
        nmf_model.fit_transform(tfidf_matrix)

        # Get the top words for the first topic
        topic_idx = 0
        top_word_indices = np.argsort(nmf_model.components_[topic_idx])[::-1][
            :n_top_words
        ]
        top_words = [str(feature_names[i]) for i in top_word_indices]

        return " ".join(top_words).title()

    def _extract_tfidf_topics(
        self, tfidf_matrix, feature_names, n_top_words: int
    ) -> str:
        """Extract topics using TF-IDF sum for small clusters.

        Args:
            tfidf_matrix: TF-IDF matrix of documents
            feature_names: Feature names from vectorizer
            n_top_words: Number of top words to include

        Returns:
            Topic label string
        """
        # Sum the TF-IDF values across all documents
        tfidf_sum = tfidf_matrix.sum(axis=0)
        # Convert to regular array for further processing
        tfidf_sum = np.asarray(tfidf_sum).flatten()
        top_indices = tfidf_sum.argsort()[::-1][:n_top_words]
        top_words = [str(feature_names[i]) for i in top_indices]

        return " ".join(top_words).title()

    def _get_topic_label_for_cluster(
        self,
        cluster_id: int,
        questions: List[str],
        use_llm: bool,
        n_topics: int,
        n_top_words: int,
        previous_topics: Optional[List[str]] = None,
    ) -> str:
        """Generate a topic label for a single cluster.

        Args:
            cluster_id: ID of the cluster
            questions: List of questions in the cluster
            use_llm: Whether to use LLM for labeling
            n_topics: Number of topics to extract
            n_top_words: Number of top words to include
            previous_topics: List of previously generated topic labels

        Returns:
            Topic label string
        """
        if not questions:
            return "Empty Cluster"

        # Try LLM-based labeling if enabled
        if use_llm:
            try:
                return self._generate_llm_topic_label(questions, previous_topics)
            except Exception as e:
                logger.warning(
                    f"Error generating LLM topic label for cluster {cluster_id}: {e}",
                    exc_info=True,
                )

        # Fall back to TF-IDF/NMF method
        try:
            return self._generate_tfidf_topic_label(questions, n_topics, n_top_words)
        except Exception as e:
            # Fallback to using the first question as the topic
            logger.warning(f"Error extracting topic for cluster {cluster_id}: {e}")
            return questions[0][:50] + "..."

    def extract_topic_labels(
        self,
        clusters: Dict[str, Any],
        n_topics: int = 1,
        n_top_words: int = 3,
        use_llm: bool = True,
        max_questions_per_cluster: int = 10,
    ) -> Dict[int, str]:
        """Extract descriptive topic labels for each cluster.

        Uses either LLM-based labeling (preferred) or TF-IDF/NMF-based keyword
        extraction (fallback) to generate meaningful labels for each cluster.

        The LLM approach produces more natural, interpretable labels, while the
        TF-IDF/NMF approach extracts statistically significant keywords.

        Args:
            clusters: Dict containing clustering results
            n_topics: Number of topics to extract per cluster (for TF-IDF/NMF method)
            n_top_words: Number of top words to include in the label
            use_llm: Whether to use an LLM for generating topic labels
            max_questions_per_cluster: Maximum number of questions to use for labeling

        Returns:
            Dict mapping cluster IDs to topic labels
        """
        # Check if LLM is available
        if use_llm and self.llm is None:
            logger.warning("LLM not provided, falling back to TF-IDF/NMF method")
            use_llm = False

        # Collect questions from each cluster
        all_cluster_questions = self._collect_cluster_questions(
            clusters, max_questions_per_cluster
        )

        # Sort clusters by size (descending) to process larger clusters first
        sorted_clusters = sorted(
            [(c["id"], len(c["source"])) for c in clusters["clusters"]],
            key=lambda x: x[1],
            reverse=True,
        )

        # Generate topic labels for each cluster, keeping track of
        # previously generated topics
        topic_labels = {}
        previous_topics = []

        for cluster_id, _ in sorted_clusters:
            questions = all_cluster_questions.get(cluster_id, [])

            topic_label = self._get_topic_label_for_cluster(
                cluster_id, questions, use_llm, n_topics, n_top_words, previous_topics
            )

            topic_labels[cluster_id] = topic_label
            previous_topics.append(topic_label)

        # Post-process to ensure uniqueness
        self._ensure_unique_labels(topic_labels)

        return topic_labels

    def _ensure_unique_labels(self, topic_labels: Dict[int, str]) -> None:
        """Ensure all topic labels are unique by adding suffixes if needed.

        Adds numeric suffixes to duplicate labels to ensure each cluster has a
        unique identifier.

        Args:
            topic_labels: Dict mapping cluster IDs to topic labels
        """
        seen_labels = {}
        seen_stems = {}  # Track semantic stems to detect similar topics

        for cluster_id, label in sorted(topic_labels.items()):
            # Check for exact duplicates
            if label in seen_labels:
                count = seen_labels[label] + 1
                seen_labels[label] = count
                topic_labels[cluster_id] = f"{label} ({count})"
            else:
                seen_labels[label] = 1

                # Check for semantic similarity with existing labels
                # This is a simple approach - just checking if one label
                # contains another
                for existing_label in seen_stems:
                    # If there's significant overlap between labels
                    overlap_condition = (
                        existing_label in label or label in existing_label
                    )
                    min_length_condition = len(label) > 5 and len(existing_label) > 5

                    if overlap_condition and min_length_condition:
                        # Add the cluster ID as a suffix to disambiguate
                        topic_labels[cluster_id] = f"{label} (C{cluster_id})"
                        break

                seen_stems[label] = cluster_id

    def generate_cluster_report(
        self,
        clusters_json_path: str,
        qa_csv_path: str,
        use_llm_for_topics: bool = True,
    ) -> pd.DataFrame:
        """Generate a comprehensive cluster quality report.

        Creates a detailed report with:
        1. Cluster sizes and IDs
        2. Topic labels for each cluster
        3. Coherence scores for each cluster
        4. Global clustering quality metrics

        The report is processed by all enabled reporters (CSV and console by default).
        Additionally, updates the original clusters JSON file with metrics.

        Args:
            clusters_json_path: Path to the JSON file containing clustering results
            qa_csv_path: Path to the CSV file containing question-answer pairs
            use_llm_for_topics: Whether to use an LLM for generating topic labels

        Returns:
            DataFrame containing the cluster report
        """
        # Load data
        clusters = self.load_clusters(clusters_json_path)
        self.qa_pairs = self.load_qa_pairs(qa_csv_path)

        # Generate embeddings
        embeddings = self.extract_embeddings_from_qa_pairs(self.qa_pairs)

        # Prepare data for metrics calculation
        embeddings_array, labels_array = self.prepare_cluster_data(clusters, embeddings)

        # Calculate global metrics
        global_metrics = self.calculate_metrics(embeddings_array, labels_array)

        # Extract topic labels
        topic_labels = self.extract_topic_labels(clusters, use_llm=use_llm_for_topics)

        # Prepare report data
        report_data = []

        # Update clusters with additional metrics
        for cluster in clusters["clusters"]:
            cluster_id = cluster["id"]
            questions = [qa["question"] for qa in cluster["source"]]

            # Calculate cluster-specific metrics
            coherence_score = self.calculate_cluster_coherence(questions)

            # Add metrics to the cluster in the original JSON
            cluster["source_count"] = len(questions)

            # Handle potential NaN values for JSON serialization
            if np.isnan(coherence_score):
                cluster["avg_similarity"] = None
                cluster["coherence_score"] = None
            else:
                cluster["avg_similarity"] = float(coherence_score)
                cluster["coherence_score"] = float(coherence_score)

            cluster["topic_label"] = topic_labels.get(cluster_id, "Unknown")

            report_data.append(
                {
                    "Cluster_ID": cluster_id,
                    "Num_QA_Pairs": len(questions),
                    "Avg_Similarity": coherence_score,
                    "Coherence_Score": coherence_score,
                    "Topic_Label": topic_labels.get(cluster_id, "Unknown"),
                }
            )

        # Add global metrics to the clusters JSON
        clusters["metrics"] = {
            "noise_ratio": float(global_metrics["noise_ratio"]),
            "davies_bouldin_score": (
                float(global_metrics.get("davies_bouldin_score", 0))
                if not np.isnan(global_metrics.get("davies_bouldin_score", np.nan))
                else None
            ),
            "calinski_harabasz_score": (
                float(global_metrics.get("calinski_harabasz_score", 0))
                if not np.isnan(global_metrics.get("calinski_harabasz_score", np.nan))
                else None
            ),
            "silhouette_score": (
                float(global_metrics.get("silhouette_score", 0))
                if not np.isnan(global_metrics.get("silhouette_score", np.nan))
                else None
            ),
        }

        # Save updated clusters JSON - read the file first to preserve any existing data
        try:
            with open(clusters_json_path, "r", encoding="utf-8") as f:
                existing_clusters_data = json.load(f)

            # Update the existing data with our new metrics
            # First update the clusters
            for i, cluster in enumerate(clusters["clusters"]):
                cluster_id = cluster["id"]
                # Find the matching cluster in the existing data
                for existing_cluster in existing_clusters_data["clusters"]:
                    if existing_cluster["id"] == cluster_id:
                        # Update with new metrics
                        existing_cluster["source_count"] = cluster["source_count"]
                        existing_cluster["avg_similarity"] = cluster["avg_similarity"]
                        existing_cluster["coherence_score"] = cluster["coherence_score"]
                        existing_cluster["topic_label"] = cluster["topic_label"]
                        break

            # Then add the global metrics
            existing_clusters_data["metrics"] = clusters["metrics"]

            # Write the updated data back to the file
            with open(clusters_json_path, "w", encoding="utf-8") as f:
                json.dump(existing_clusters_data, f, indent=2)

            logger.debug(
                f"Updated existing clusters JSON with metrics: {clusters_json_path}"
            )
        except Exception as e:
            logger.warning(
                f"Error updating existing clusters JSON, creating new file: {e}"
            )
            # Fallback to creating a new file if there's an error
            with open(clusters_json_path, "w", encoding="utf-8") as f:
                json.dump(clusters, f, indent=2)
            logger.debug(
                f"Created new clusters JSON with metrics: {clusters_json_path}"
            )

        # Create DataFrame
        report_df = pd.DataFrame(report_data)

        # Add global metrics as a summary row
        summary_metrics = (
            f"Noise Ratio: {global_metrics['noise_ratio']:.2f}, "
            f"DB: {global_metrics.get('davies_bouldin_score', np.nan):.2f}, "
            f"CH: {global_metrics.get('calinski_harabasz_score', np.nan):.2f}"
        )

        summary_row = pd.DataFrame(
            [
                {
                    "Cluster_ID": "SUMMARY",
                    "Num_QA_Pairs": len(self.qa_pairs),
                    "Avg_Similarity": np.nan,
                    "Coherence_Score": np.nan,
                    "Topic_Label": summary_metrics,
                }
            ]
        )

        report_df = pd.concat([report_df, summary_row], ignore_index=True)

        # Get top clusters by size
        top_clusters = report_df[report_df["Cluster_ID"] != "SUMMARY"].nlargest(
            5, "Num_QA_Pairs"
        )

        # Create report data object
        report_data_obj = ReportData(
            report_df=report_df,
            clusters_json_path=clusters_json_path,
            output_dir=self.output_dir,
            summary_metrics=global_metrics,
            top_clusters=top_clusters,
        )

        # Generate reports using all enabled reporters
        self.reporter_registry.generate_reports(report_data_obj)

        return report_df

"""Unit tests for the evaluation module."""

from unittest.mock import patch

import numpy as np
import pytest

from clusx.evaluation import ClusterEvaluator


@pytest.fixture
def embeddings_fx():
    """Return sample embeddings for testing."""
    return np.array(
        [
            [1.0, 0.0, 0.0],  # Cluster 0
            [0.9, 0.1, 0.0],  # Cluster 0
            [0.0, 1.0, 0.0],  # Cluster 1
            [0.1, 0.9, 0.0],  # Cluster 1
            [0.0, 0.0, 1.0],  # Cluster 2 (singleton)
        ]
    )


@pytest.fixture
def texts_fx():
    """Return sample texts for testing."""
    return [
        "Text in cluster 0",
        "Another text in cluster 0",
        "Text in cluster 1",
        "Another text in cluster 1",
        "Singleton text in cluster 2",
    ]


@pytest.fixture
def cluster_assignments_fx():
    """Return sample cluster assignments for testing."""
    return [0, 0, 1, 1, 2]


@pytest.fixture
def evaluator_fx(texts_fx, embeddings_fx, cluster_assignments_fx):
    """Return a ClusterEvaluator instance with sample data."""
    return ClusterEvaluator(
        texts=texts_fx,
        embeddings=embeddings_fx,
        cluster_assignments=cluster_assignments_fx,
        model_name="TestModel",
        alpha=1.0,
        sigma=0.0,
        kappa=0.0,
        random_state=42,
    )


def test_silhouette_score_with_valid_clusters(evaluator_fx):
    """Test silhouette score calculation with valid clusters."""
    with patch(
        "clusx.evaluation.silhouette_score", return_value=0.75
    ) as silhouette_mock:
        score = evaluator_fx.calculate_silhouette_score()

        assert score == 0.75

        # Verify silhouette_score was called with filtered data
        # (excluding singleton cluster)
        args, kwargs = silhouette_mock.call_args
        filtered_embeddings, filtered_assignments = args

        # Should only include clusters 0 and 1 (4 samples total)
        assert len(filtered_embeddings) == 4
        assert len(filtered_assignments) == 4
        assert set(filtered_assignments) == {0, 1}
        assert kwargs["metric"] == "cosine"


def test_silhouette_score_with_only_singletons():
    """Test silhouette score calculation when all clusters are singletons."""
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cluster_assignments = [0, 1, 2]  # Each text in its own cluster

    evaluator = ClusterEvaluator(
        texts=texts,
        embeddings=embeddings,
        cluster_assignments=cluster_assignments,
        model_name="SingletonModel",
        alpha=1.0,
        sigma=0.0,
        kappa=0.0,
    )

    # Should return 0.0 as there are no valid clusters with ≥2 samples
    with patch("clusx.evaluation.logger.warning") as warning_mock:
        score = evaluator.calculate_silhouette_score()
        assert score == 0.0
        warning_mock.assert_called_once()
        assert "fewer than 2 valid clusters found" in warning_mock.call_args[0][0]


def test_silhouette_score_with_fewer_than_two_clusters():
    """Test silhouette score calculation with fewer than two clusters."""
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
    cluster_assignments = [0, 0, 0]  # All texts in the same cluster

    evaluator = ClusterEvaluator(
        texts=texts,
        embeddings=embeddings,
        cluster_assignments=cluster_assignments,
        model_name="SingleClusterModel",
        alpha=1.0,
        sigma=0.0,
        kappa=0.0,
    )

    # Should return 0.0 as there's only one cluster
    with patch("clusx.evaluation.logger.warning") as warning_mock:
        score = evaluator.calculate_silhouette_score()
        assert score == 0.0
        warning_mock.assert_called_once()
        assert "fewer than 2 valid clusters found" in warning_mock.call_args[0][0]


def test_silhouette_score_with_exception():
    """Test silhouette score calculation when an exception occurs."""
    texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])
    cluster_assignments = [0, 0, 1, 1]

    evaluator = ClusterEvaluator(
        texts=texts,
        embeddings=embeddings,
        cluster_assignments=cluster_assignments,
        model_name="ExceptionModel",
        alpha=1.0,
        sigma=0.0,
        kappa=0.0,
    )

    with patch(
        "clusx.evaluation.silhouette_score", side_effect=ValueError("Test error")
    ):
        with patch("clusx.evaluation.logger.error") as error_mock:
            score = evaluator.calculate_silhouette_score()
            assert score == 0.0
            error_mock.assert_called_once()
            assert "Error calculating silhouette score" in error_mock.call_args[0][0]


def test_silhouette_score_with_mixed_valid_and_singleton():
    """Test silhouette score with a mix of valid clusters and singletons."""
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5", "Text 6"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # Cluster 0
            [0.9, 0.1, 0.0],  # Cluster 0
            [0.0, 1.0, 0.0],  # Cluster 1
            [0.1, 0.9, 0.0],  # Cluster 1
            [0.0, 0.0, 1.0],  # Cluster 2 (singleton)
            [0.5, 0.5, 0.0],  # Cluster 3 (singleton)
        ]
    )
    cluster_assignments = [0, 0, 1, 1, 2, 3]

    evaluator = ClusterEvaluator(
        texts=texts,
        embeddings=embeddings,
        cluster_assignments=cluster_assignments,
        model_name="MixedModel",
        alpha=1.0,
        sigma=0.0,
        kappa=0.0,
    )

    with patch(
        "clusx.evaluation.silhouette_score", return_value=0.8
    ) as silhouette_mock:
        score = evaluator.calculate_silhouette_score()

        # Verify the score matches our mock
        assert score == 0.8

        # Verify silhouette_score was called with filtered data
        # (excluding singleton clusters)
        args, _ = silhouette_mock.call_args
        filtered_embeddings, filtered_assignments = args

        # Should only include clusters 0 and 1 (4 samples total)
        assert len(filtered_embeddings) == 4
        assert len(filtered_assignments) == 4
        assert set(filtered_assignments) == {0, 1}


def test_silhouette_score_logging(evaluator_fx):
    """Test that silhouette score calculation logs appropriate information."""
    with patch("clusx.evaluation.silhouette_score", return_value=0.75):
        with patch("clusx.evaluation.logger.info") as info_mock:
            evaluator_fx.calculate_silhouette_score()

            info_mock.assert_called_once()

            expected = (
                "Silhouette score for %s: %.4f (using %d/%d samples in %d/%d clusters)"
            )
            actual, *args = info_mock.call_args[0]
            assert actual == expected

            # Check the arguments passed to the logger
            (
                model_name,
                score,
                valid_samples,
                total_samples,
                valid_clusters,
                total_clusters,
            ) = args
            assert model_name == "TestModel"
            assert score == 0.75
            assert valid_samples == 4  # 4 samples in valid clusters
            assert total_samples == 5  # 5 total samples
            assert valid_clusters == 2  # 2 valid clusters
            assert total_clusters == 3  # 3 total clusters

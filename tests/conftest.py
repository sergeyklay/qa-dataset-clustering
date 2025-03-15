"""Common test fixtures, mocks and configurations."""

import csv
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clusx.clustering.models import DirichletProcess, PitmanYorProcess


@pytest.fixture
def embedding_fx() -> np.ndarray:
    """Return a sample embedding numpy array."""
    return np.asarray([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def transformer_mock(
    embedding_fx: np.ndarray, patch_sentence_transformer
):  # pylint: disable=redefined-outer-name
    """
    Configure the mock SentenceTransformer for specific tests.

    This fixture uses the globally patched SentenceTransformer and configures
    it to return the embedding_fx when encode is called.
    """
    mock_instance = patch_sentence_transformer.return_value
    mock_instance.encode.return_value = embedding_fx
    return patch_sentence_transformer


@pytest.fixture(autouse=True)
def patch_sentence_transformer():
    """
    Patch SentenceTransformer to prevent model loading in all tests.

    This fixture is automatically used in all tests to prevent the actual
    SentenceTransformer model from being loaded, which significantly
    improves test performance.
    """
    with patch("clusx.clustering.models.SentenceTransformer") as mock_st:
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance
        yield mock_st


@pytest.fixture
def dp_instance():
    """Return a pre-configured DirichletProcess instance."""
    return DirichletProcess(alpha=1.0)


@pytest.fixture
def pyp_instance():
    """Return a pre-configured PitmanYorProcess instance."""
    return PitmanYorProcess(alpha=1.0, sigma=0.5)


@pytest.fixture
def basic_qa_csv(tmp_path):
    """Create a basic CSV file with question-answer pairs."""
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerow(["What is Python?", "Python is a programming language."])
        writer.writerow(
            ["What is TensorFlow?", "TensorFlow is a machine learning framework."]
        )
    return csv_path


@pytest.fixture
def custom_columns_csv(tmp_path):
    """Create a CSV file with custom column names."""
    csv_path = tmp_path / "test_custom.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "response"])
        writer.writerow(["What is Python?", "Python is a programming language."])
        writer.writerow(
            ["What is TensorFlow?", "TensorFlow is a machine learning framework."]
        )
    return csv_path


@pytest.fixture
def csv_with_empty_rows(tmp_path):
    """Create a CSV file with empty rows."""
    csv_path = tmp_path / "test_empty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerow(["What is Python?", "Python is a programming language."])
        writer.writerow(["", "This row should be skipped."])
        writer.writerow(
            ["What is TensorFlow?", "TensorFlow is a machine learning framework."]
        )
    return csv_path


@pytest.fixture
def basic_text_file(tmp_path):
    """Create a basic text file with one text per line."""
    text_path = tmp_path / "test.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("What is Python?\n")
        f.write("What is TensorFlow?\n")
        f.write("\n")  # Empty line should be skipped
        f.write("What is PyTorch?\n")
    return text_path


@pytest.fixture
def sample_texts():
    """Return a list of sample question texts."""
    return ["What is Python?", "What is TensorFlow?", "What is PyTorch?"]


@pytest.fixture
def sample_clusters():
    """Return a list of sample cluster assignments."""
    return [0, 1, 1]


@pytest.fixture
def sample_data():
    """Return a list of sample data dictionaries."""
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a programming language.",
        },
        {
            "question": "What is TensorFlow?",
            "answer": "TensorFlow is a machine learning framework.",
        },
        {
            "question": "What is PyTorch?",
            "answer": "PyTorch is another machine learning framework.",
        },
    ]


@pytest.fixture
def cluster_assignments_csv(tmp_path):
    """Create a CSV file with cluster assignments."""
    csv_path = tmp_path / "cluster_assignments.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "Cluster_DP", "Alpha", "Sigma", "Kappa"])
        writer.writerow(["What is Python?", "0", "1.0", "0.0", "0.1"])
        writer.writerow(["What is TensorFlow?", "1", "1.0", "0.0", "0.1"])
        writer.writerow(["What is PyTorch?", "1", "1.0", "0.0", "0.1"])
    return csv_path


@pytest.fixture
def cluster_assignments_custom_column_csv(tmp_path):
    """Create a CSV file with cluster assignments using a custom column name."""
    csv_path = tmp_path / "cluster_assignments_custom.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "Cluster_Custom", "Alpha", "Sigma", "Kappa"])
        writer.writerow(["What is Python?", "0", "1.0", "0.0", "0.1"])
        writer.writerow(["What is TensorFlow?", "1", "1.0", "0.0", "0.1"])
        writer.writerow(["What is PyTorch?", "1", "1.0", "0.0", "0.1"])
    return csv_path


@pytest.fixture
def cluster_assignments_no_cluster_column_csv(tmp_path):
    """Create a CSV file without a cluster column."""
    csv_path = tmp_path / "no_cluster_column.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "OtherColumn"])
        writer.writerow(["What is Python?", "value1"])
        writer.writerow(["What is TensorFlow?", "value2"])
        writer.writerow(["What is PyTorch?", "value3"])
    return csv_path

"""Tests for the benchmarker module."""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from qadst.benchmarker import ClusterBenchmarker
from qadst.reporters import ConsoleReporter, CSVReporter


class TestClusterBenchmarker:
    """Tests for the ClusterBenchmarker class."""

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    @patch("qadst.benchmarker.ReporterRegistry")
    def test_init_default_values(
        self, mock_registry, mock_chat_openai, mock_embeddings
    ):
        """Test initialization with default values."""
        # Setup mock registry
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        # Create benchmarker with default values
        benchmarker = ClusterBenchmarker()

        # Check default values
        assert benchmarker.output_dir == "./output"
        assert benchmarker.embeddings_model is None
        assert benchmarker.llm is None

        # Check that the reporter registry was initialized
        assert mock_registry.called
        assert benchmarker.reporter_registry == mock_registry_instance

        # Check that the reporters were registered
        assert mock_registry_instance.register.call_count == 2

        # Verify the first call was to register the CSV reporter
        args, kwargs = mock_registry_instance.register.call_args_list[0]
        assert args[0] == "csv"
        assert isinstance(args[1], CSVReporter)
        assert kwargs.get("enabled", False) is True

        # Verify the second call was to register the console reporter
        args, kwargs = mock_registry_instance.register.call_args_list[1]
        assert args[0] == "console"
        assert isinstance(args[1], ConsoleReporter)
        assert kwargs.get("enabled", False) is True

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_init_with_models(self, mock_chat_openai, mock_embeddings):
        """Test initialization with model names."""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create benchmarker with custom values
            benchmarker = ClusterBenchmarker(
                embedding_model_name="text-embedding-3-large",
                llm_model_name="gpt-4o",
                output_dir=temp_dir,
            )

            # Check values
            assert benchmarker.output_dir == temp_dir
            mock_embeddings.assert_called_once()
            assert (
                mock_embeddings.call_args.kwargs.get("model")
                == "text-embedding-3-large"
            )

            mock_chat_openai.assert_called_once()
            assert mock_chat_openai.call_args.kwargs.get("model") == "gpt-4o"
            assert mock_chat_openai.call_args.kwargs.get("temperature") == 0

            # Verify the output directory was created
            assert os.path.exists(temp_dir)

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    @patch("qadst.benchmarker.logger")
    def test_init_with_model_errors(
        self, mock_logger, mock_chat_openai, mock_embeddings
    ):
        """Test initialization with model errors."""
        # Setup mocks to raise exceptions
        mock_embeddings.side_effect = Exception("Embedding model error")
        mock_chat_openai.side_effect = Exception("LLM error")

        # Create benchmarker with models that will fail
        benchmarker = ClusterBenchmarker(
            embedding_model_name="invalid-model", llm_model_name="invalid-llm"
        )

        # Check that models are None due to errors
        assert benchmarker.embeddings_model is None
        assert benchmarker.llm is None

        # Verify warning logs were created
        assert mock_logger.warning.call_count == 2
        mock_logger.warning.assert_any_call(
            "Failed to initialize embeddings model: Embedding model error"
        )
        mock_logger.warning.assert_any_call("Failed to initialize LLM: LLM error")

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_clusters_valid_json(self, mock_chat_openai, mock_embeddings):
        """Test loading clusters from a valid JSON file."""
        # Create test data
        test_clusters = {
            "clusters": [
                {
                    "id": 1,
                    "representative": [{"question": "Test Q1", "answer": "Test A1"}],
                    "source": [
                        {"question": "Test Q1", "answer": "Test A1"},
                        {"question": "Test Q2", "answer": "Test A2"},
                    ],
                },
                {
                    "id": 2,
                    "representative": [{"question": "Test Q3", "answer": "Test A3"}],
                    "source": [{"question": "Test Q3", "answer": "Test A3"}],
                },
            ]
        }

        # Create a temporary file with test data
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(test_clusters, f)
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Load clusters
            loaded_clusters = benchmarker.load_clusters(temp_file_name)

            # Verify the loaded data
            assert loaded_clusters == test_clusters
            assert len(loaded_clusters["clusters"]) == 2
            assert loaded_clusters["clusters"][0]["id"] == 1
            assert loaded_clusters["clusters"][1]["id"] == 2
            assert len(loaded_clusters["clusters"][0]["source"]) == 2
            assert len(loaded_clusters["clusters"][1]["source"]) == 1

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_clusters_invalid_json(self, mock_chat_openai, mock_embeddings):
        """Test loading clusters from an invalid JSON file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            f.write("This is not valid JSON")
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Attempt to load clusters from invalid JSON
            with patch("qadst.benchmarker.json.load") as mock_json_load:
                mock_json_load.side_effect = json.JSONDecodeError("Test error", "", 0)

                # Should raise JSONDecodeError
                try:
                    benchmarker.load_clusters(temp_file_name)
                    assert False, "Expected JSONDecodeError was not raised"
                except json.JSONDecodeError:
                    pass  # Test passed

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_clusters_file_not_found(self, mock_chat_openai, mock_embeddings):
        """Test loading clusters from a non-existent file."""
        # Create benchmarker
        benchmarker = ClusterBenchmarker()

        # Attempt to load clusters from non-existent file
        non_existent_file = "/path/to/non/existent/file.json"

        # Should raise FileNotFoundError
        try:
            benchmarker.load_clusters(non_existent_file)
            assert False, "Expected FileNotFoundError was not raised"
        except FileNotFoundError:
            pass  # Test passed

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_qa_pairs_valid_csv(self, mock_chat_openai, mock_embeddings):
        """Test loading QA pairs from a valid CSV file."""
        # Create test data
        test_qa_pairs = [
            ("How do I reset my password?", "Click the 'Forgot Password' link."),
            (
                "What payment methods do you accept?",
                "We accept credit cards and PayPal.",
            ),
            ("How do I contact support?", "Email us at support@example.com."),
        ]

        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])  # Header
            for q, a in test_qa_pairs:
                writer.writerow([q, a])
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Load QA pairs
            loaded_qa_pairs = benchmarker.load_qa_pairs(temp_file_name)

            # Verify the loaded data
            assert len(loaded_qa_pairs) == len(test_qa_pairs)
            for i, (q, a) in enumerate(test_qa_pairs):
                assert loaded_qa_pairs[i][0] == q
                assert loaded_qa_pairs[i][1] == a

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_qa_pairs_empty_csv(self, mock_chat_openai, mock_embeddings):
        """Test loading QA pairs from an empty CSV file."""
        # Create a temporary CSV file with only a header
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])  # Header only
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Load QA pairs
            loaded_qa_pairs = benchmarker.load_qa_pairs(temp_file_name)

            # Verify the loaded data is empty
            assert len(loaded_qa_pairs) == 0

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_qa_pairs_malformed_csv(self, mock_chat_openai, mock_embeddings):
        """Test loading QA pairs from a malformed CSV file."""
        # Create a temporary CSV file with some rows having fewer than 2 columns
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])  # Header
            writer.writerow(
                ["How do I reset my password?", "Click the 'Forgot Password' link."]
            )
            writer.writerow(["Incomplete row"])  # Only one column
            writer.writerow(
                [
                    "What payment methods do you accept?",
                    "We accept credit cards and PayPal.",
                ]
            )
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Load QA pairs
            loaded_qa_pairs = benchmarker.load_qa_pairs(temp_file_name)

            # Verify the loaded data (should skip the incomplete row)
            assert len(loaded_qa_pairs) == 2
            assert loaded_qa_pairs[0][0] == "How do I reset my password?"
            assert loaded_qa_pairs[1][0] == "What payment methods do you accept?"

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_qa_pairs_file_not_found(self, mock_chat_openai, mock_embeddings):
        """Test loading QA pairs from a non-existent file."""
        # Create benchmarker
        benchmarker = ClusterBenchmarker()

        # Attempt to load QA pairs from non-existent file
        non_existent_file = "/path/to/non/existent/file.csv"

        # Should raise FileNotFoundError
        try:
            benchmarker.load_qa_pairs(non_existent_file)
            assert False, "Expected FileNotFoundError was not raised"
        except FileNotFoundError:
            pass  # Test passed

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_extract_embeddings_from_qa_pairs_success(
        self, mock_chat, mock_openai_embeddings
    ):
        """Test extracting embeddings from QA pairs successfully."""
        # Create test data
        test_qa_pairs = [
            ("How do I reset my password?", "Click the 'Forgot Password' link."),
            (
                "What payment methods do you accept?",
                "We accept credit cards and PayPal.",
            ),
            ("How do I contact support?", "Email us at support@example.com."),
        ]

        # Setup mock embeddings model
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance

        # Mock the embed_documents method to return test embeddings
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_embeddings_instance.embed_documents.return_value = test_embeddings

        # Create benchmarker with mock embeddings model
        benchmarker = ClusterBenchmarker(embedding_model_name="test-model")

        # Extract embeddings
        embeddings = benchmarker.extract_embeddings_from_qa_pairs(test_qa_pairs)

        # Verify the embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 3)  # 3 questions, 3 dimensions
        assert np.array_equal(embeddings, np.array(test_embeddings))

        # Verify the embed_documents method was called with the correct questions
        expected_questions = [q for q, _ in test_qa_pairs]
        mock_embeddings_instance.embed_documents.assert_called_once_with(
            expected_questions
        )

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_extract_embeddings_from_qa_pairs_no_model(
        self, mock_chat, mock_openai_embeddings
    ):
        """Test extracting embeddings without an embeddings model."""
        # Create test data
        test_qa_pairs = [
            ("How do I reset my password?", "Click the 'Forgot Password' link."),
            (
                "What payment methods do you accept?",
                "We accept credit cards and PayPal.",
            ),
        ]

        # Create benchmarker without an embeddings model
        benchmarker = ClusterBenchmarker()
        benchmarker.embeddings_model = None

        # Attempt to extract embeddings without a model
        try:
            benchmarker.extract_embeddings_from_qa_pairs(test_qa_pairs)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Embedding model name not provided"

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_extract_embeddings_from_qa_pairs_empty_list(
        self, mock_chat, mock_openai_embeddings
    ):
        """Test extracting embeddings from an empty list of QA pairs."""
        # Setup mock embeddings model
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance

        # Mock the embed_documents method to return an empty list
        mock_embeddings_instance.embed_documents.return_value = []

        # Create benchmarker with mock embeddings model
        benchmarker = ClusterBenchmarker(embedding_model_name="test-model")

        # Extract embeddings from an empty list
        embeddings = benchmarker.extract_embeddings_from_qa_pairs([])

        # Verify the embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)  # Empty array

        # Verify the embed_documents method was called with an empty list
        mock_embeddings_instance.embed_documents.assert_called_once_with([])

    @patch("qadst.embeddings.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_extract_embeddings_from_qa_pairs_model_error(
        self, mock_chat, mock_openai_embeddings
    ):
        """Test handling errors from the embeddings model."""
        # Create test data
        test_qa_pairs = [
            ("How do I reset my password?", "Click the 'Forgot Password' link."),
            (
                "What payment methods do you accept?",
                "We accept credit cards and PayPal.",
            ),
        ]

        # Setup mock embeddings model
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance

        # Mock the embed_documents method to raise an exception
        mock_embeddings_instance.embed_documents.side_effect = Exception(
            "Embedding error"
        )

        # Create benchmarker with mock embeddings model
        benchmarker = ClusterBenchmarker(embedding_model_name="test-model")

        # Attempt to extract embeddings with a failing model
        try:
            benchmarker.extract_embeddings_from_qa_pairs(test_qa_pairs)
            assert False, "Expected Exception was not raised"
        except Exception as e:
            assert str(e) == "Embedding error"

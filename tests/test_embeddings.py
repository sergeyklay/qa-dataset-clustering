"""Tests for the embeddings module."""

import unittest
from unittest.mock import patch

from qadst.embeddings import get_embeddings_model


class TestEmbeddings(unittest.TestCase):
    """Test the embeddings module."""

    @patch("qadst.embeddings.SbertEmbeddings")
    def test_get_embeddings_model_sbert(self, mock_sbert):
        """Test get_embeddings_model with SBERT."""
        # Test with 'sbert'
        get_embeddings_model("sbert")
        mock_sbert.assert_called_once()

        # Reset mock
        mock_sbert.reset_mock()

        # Test with 'sbert:custom-model'
        get_embeddings_model("sbert:custom-model")
        mock_sbert.assert_called_once_with(model_name="custom-model")

        # Reset mock
        mock_sbert.reset_mock()

        # Test with 'sentence-transformers'
        get_embeddings_model("sentence-transformers/all-MiniLM-L6-v2")
        mock_sbert.assert_called_once()

    @patch("qadst.embeddings.OpenAIEmbeddings")
    def test_get_embeddings_model_openai(self, mock_openai):
        """Test get_embeddings_model with OpenAI."""
        # Test with OpenAI model
        get_embeddings_model("text-embedding-3-large")
        mock_openai.assert_called_once()

        # Test with API key
        mock_openai.reset_mock()
        get_embeddings_model("text-embedding-3-large", openai_api_key="test-key")
        mock_openai.assert_called_once()


if __name__ == "__main__":
    unittest.main()

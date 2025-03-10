"""Embedding models for qadst."""

import logging
import os
from typing import Any, List, Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class SbertEmbeddings:
    """Sentence-BERT embeddings model.

    This class provides an interface compatible with OpenAIEmbeddings
    but uses the Sentence-BERT model for generating embeddings locally.

    Attributes:
        model: The Sentence-BERT model instance
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the SBERT embeddings model.

        Args:
            model_name: Name of the SBERT model to use
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "Could not import sentence-transformers. "
                "Please install it with `pip install sentence-transformers`."
            )
        logger.info(f"Initialized SBERT embeddings model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[Any]:
        """Embed a list of documents/texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, each embedding is a list of floats
        """
        if not texts:
            return []

        # Convert to numpy array and then to list for consistency with OpenAIEmbeddings
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding texts with SBERT: {e}")
            # Return empty list in case of error
            return []


def get_embeddings_model(model_name: str, openai_api_key: Optional[str] = None) -> Any:
    """Factory function to get the appropriate embeddings model.

    Args:
        model_name: Name of the embedding model to use
        openai_api_key: OpenAI API key (only needed for OpenAI models)

    Returns:
        An embeddings model instance
    """
    if "sbert" in model_name.lower() or "sentence-transformers" in model_name.lower():
        # Extract the actual model name if it's in the format "sbert:model_name"
        if ":" in model_name:
            _, actual_model = model_name.split(":", 1)
            return SbertEmbeddings(model_name=actual_model)
        return SbertEmbeddings()

    # Default to OpenAI embeddings
    # Use environment variable if no API key is provided
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    # Create OpenAI embeddings with the correct parameter name
    return OpenAIEmbeddings(model=model_name, api_key=api_key)

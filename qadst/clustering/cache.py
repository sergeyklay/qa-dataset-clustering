"""
Embedding cache implementation for efficient text embedding storage and retrieval.
"""

import hashlib
import os
import pickle
from typing import Optional

from torch import Tensor

from qadst.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Provides caching functionality for embeddings with class-specific cache files."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.embedding_cache: dict[str, Tensor] = {}

    def get_cache_path(self, base_name: str):
        """Generate cache file path based on class name."""
        if not self.cache_dir:
            return None

        base_name = base_name.lower().translate(str.maketrans(" ./", "___"))
        return os.path.join(self.cache_dir, f"{base_name}.pkl")

    def load_cache(self, base_name: Optional[str] = "embeddings") -> dict[str, Tensor]:
        """Load cache for the specified class."""
        if not self.cache_dir:
            return {}

        base_name = base_name or "embeddings"
        cache_path = self.get_cache_path(base_name)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {e}")
                self.embedding_cache = {}
        return self.embedding_cache

    def save_cache(self, base_name: Optional[str] = "embeddings"):
        """Save cache for the specified class."""
        if not self.cache_dir or not self.embedding_cache:
            return

        base_name = base_name or "embeddings"
        cache_path = self.get_cache_path(base_name)
        if cache_path:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")

    def _hash_key(self, key) -> str:
        """Calculate a stable hash for an items list."""
        combined = str(key).encode("utf-8")

        # SHA-256 produces 64-character hex digest
        return hashlib.sha256(combined).hexdigest()

    def get(self, key: str) -> Tensor | None:
        """Get item from cache."""
        return self.embedding_cache.get(self._hash_key(key))

    def set(self, key: str, value: Tensor):
        """Set item in cache."""
        self.embedding_cache[self._hash_key(key)] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._hash_key(key) in self.embedding_cache

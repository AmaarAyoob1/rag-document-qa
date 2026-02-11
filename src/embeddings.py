"""
Embedding model wrapper for text vectorization.
Uses local HuggingFace sentence-transformers for privacy and speed.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from chunker import Chunk


class EmbeddingModel:
    """Wrapper around sentence-transformers for consistent embedding interface."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name. Good options:
                - "all-MiniLM-L6-v2" (fast, 384 dim, good quality)
                - "all-mpnet-base-v2" (slower, 768 dim, best quality)
            device: "cpu" or "cuda"
            normalize: Whether to L2-normalize embeddings (required for cosine similarity)
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name} (dim={self.dimension})")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of text strings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
        
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
        """Embed a list of Chunk objects."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size)

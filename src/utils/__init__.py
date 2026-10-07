"""Utility functions and helpers"""

from src.utils.embeddings import (
    get_embedding,
    get_embeddings_batch,
    cosine_similarity,
    cosine_similarity_matrix,
)

__all__ = [
    "get_embedding",
    "get_embeddings_batch",
    "cosine_similarity",
    "cosine_similarity_matrix",
]

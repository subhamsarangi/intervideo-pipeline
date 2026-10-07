"""Embedding utilities for semantic similarity and ranking using OpenAI embeddings"""

import os
from typing import List, Optional, Union
from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI, OpenAI

from src.utils.llm_tracking import increment_llm_calls

# Load environment variables
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return api_key


def get_async_openai_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client instance"""
    return AsyncOpenAI(api_key=get_openai_api_key())


def get_sync_openai_client() -> OpenAI:
    """Get sync OpenAI client instance"""
    return OpenAI(api_key=get_openai_api_key())


async def get_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: Optional[int] = None,
    client: Optional[AsyncOpenAI] = None,
) -> List[float]:
    """
    Generate embedding for a single text string.

    Args:
        text: Text to embed
        model: Embedding model name (default: text-embedding-3-small)
        dimensions: Optional output dimension size
        client: Optional AsyncOpenAI client instance

    Returns:
        Embedding vector as list of floats
    """
    if not text or not text.strip():
        text = " "

    # Clean text
    clean_text = text.replace("\r\n", " ").replace("\n", " ").strip()
    if not clean_text:
        clean_text = " "

    cli = client or get_async_openai_client()

    kwargs = {"input": clean_text, "model": model}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    response = await cli.embeddings.create(**kwargs)
    if hasattr(response, "usage") and response.usage:
        increment_llm_calls(
            operation="embedding_single",
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=0,
        )
    return response.data[0].embedding


async def get_embeddings_batch(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: Optional[int] = None,
    batch_size: int = 100,
    client: Optional[AsyncOpenAI] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts in batches.

    Args:
        texts: List of text strings to embed
        model: Embedding model name (default: text-embedding-3-small)
        dimensions: Optional output dimension size
        batch_size: Maximum texts per OpenAI request (default: 100)
        client: Optional AsyncOpenAI client instance

    Returns:
        List of embedding vectors preserving input order
    """
    if not texts:
        return []

    cli = client or get_async_openai_client()
    all_embeddings: List[List[float]] = []

    # Clean texts
    cleaned_texts = [
        (t.replace("\r\n", " ").replace("\n", " ").strip() if t and t.strip() else " ")
        for t in texts
    ]

    for i in range(0, len(cleaned_texts), batch_size):
        chunk = cleaned_texts[i : i + batch_size]
        kwargs = {"input": chunk, "model": model}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = await cli.embeddings.create(**kwargs)
        if hasattr(response, "usage") and response.usage:
            increment_llm_calls(
                operation="embedding_batch",
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0,
            )
        # Sort by index to ensure order is preserved
        chunk_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(chunk_embeddings)

    return all_embeddings


def cosine_similarity(
    vec_a: Union[List[float], np.ndarray],
    vec_b: Union[List[float], np.ndarray],
) -> float:
    """
    Compute cosine similarity between two 1D vectors.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Cosine similarity score in range [-1.0, 1.0] (typically [0.0, 1.0] for text embeddings)
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = np.dot(a, b)
    similarity = dot / (norm_a * norm_b)
    # Clip to handle minor floating point inaccuracies
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_similarity_matrix(
    matrix_a: Union[List[List[float]], np.ndarray],
    matrix_b: Union[List[List[float]], np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between two sets of vectors.

    Args:
        matrix_a: Shape (N, D) array of vectors
        matrix_b: Shape (M, D) array of vectors

    Returns:
        Similarity matrix of shape (N, M) where [i, j] is similarity between A[i] and B[j]
    """
    a = np.asarray(matrix_a, dtype=np.float32)
    b = np.asarray(matrix_b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero
    norm_a[norm_a == 0] = 1.0
    norm_b[norm_b == 0] = 1.0

    normalized_a = a / norm_a
    normalized_b = b / norm_b

    similarity_matrix = np.dot(normalized_a, normalized_b.T)
    return np.clip(similarity_matrix, -1.0, 1.0)

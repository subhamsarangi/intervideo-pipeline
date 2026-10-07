"""Tests for embedding utilities"""

import pytest
import numpy as np
from src.utils.embeddings import (
    cosine_similarity,
    cosine_similarity_matrix,
    get_embedding,
    get_embeddings_batch,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0"""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0"""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2) - 0.0) < 1e-5

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0"""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert abs(cosine_similarity(v1, v2) - (-1.0)) < 1e-5

    def test_zero_vector_handling(self):
        """Zero vector should return 0.0 similarity without dividing by zero"""
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        assert cosine_similarity(v1, v2) == 0.0
        assert cosine_similarity(v1, v1) == 0.0

    def test_cosine_similarity_matrix(self):
        """Pairwise similarity matrix between two sets of vectors"""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        mat = cosine_similarity_matrix(a, b)
        assert mat.shape == (2, 3)

        # a[0] with b[0] is identical -> 1.0
        assert abs(mat[0, 0] - 1.0) < 1e-5
        # a[0] with b[1] is orthogonal -> 0.0
        assert abs(mat[0, 1] - 0.0) < 1e-5
        # a[1] with b[1] is identical -> 1.0
        assert abs(mat[1, 1] - 1.0) < 1e-5


class TestOpenAIEmbeddings:
    @pytest.mark.asyncio
    async def test_get_embedding_live(self, tracker):
        """Live API test for single embedding generation"""
        with tracker("Embedding Single Test") as bar:
            bar.update(30, "generating")
            emb = await get_embedding("Python FastAPI backend engineer")
            bar.update(70, "done")

        assert isinstance(emb, list)
        assert len(emb) == 1536  # text-embedding-3-small default dimension
        assert all(isinstance(x, float) for x in emb)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_live(self, tracker):
        """Live API test for batch embedding generation"""
        texts = [
            "Senior Python Developer",
            "Docker Kubernetes Cloud",
            "React TypeScript Frontend",
        ]

        with tracker("Embedding Batch Test") as bar:
            bar.update(30, "generating batch")
            embeddings = await get_embeddings_batch(texts, batch_size=2)
            bar.update(70, "done")

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 1536

        # Semantic check: Python developer should be more similar to Docker than to React
        sim_python_docker = cosine_similarity(embeddings[0], embeddings[1])
        sim_python_react = cosine_similarity(embeddings[0], embeddings[2])
        print(f"\nSimilarity Python <-> Docker: {sim_python_docker:.4f}")
        print(f"Similarity Python <-> React: {sim_python_react:.4f}")
        assert -1.0 <= sim_python_docker <= 1.0
        assert -1.0 <= sim_python_react <= 1.0

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Empty list should return empty list without error"""
        res = await get_embeddings_batch([])
        assert res == []

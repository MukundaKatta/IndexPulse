"""Tests for indexpulse core functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from indexpulse import VectorIndex, IndexConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_vec(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Test: add and search vectors
# ---------------------------------------------------------------------------


class TestAddAndSearch:
    def test_add_and_search_cosine(self) -> None:
        index = VectorIndex(dimension=64, default_metric="cosine")

        vecs = {f"v{i}": _random_vec() for i in range(20)}
        for vid, vec in vecs.items():
            index.add(id=vid, vector=vec, metadata={"i": vid})

        assert len(index) == 20

        # Search with the first vector — it should be the top result
        query = vecs["v0"]
        results = index.search(query, k=5)

        assert len(results) == 5
        assert results[0].id == "v0"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[0].metadata == {"i": "v0"}

    def test_add_duplicate_raises(self) -> None:
        index = VectorIndex(dimension=4)
        index.add("a", [1, 2, 3, 4])
        with pytest.raises(ValueError, match="already exists"):
            index.add("a", [5, 6, 7, 8])

    def test_delete_and_contains(self) -> None:
        index = VectorIndex(dimension=4)
        index.add("x", [1, 0, 0, 0])
        assert "x" in index
        index.delete("x")
        assert "x" not in index
        assert len(index) == 0

    def test_update_vector(self) -> None:
        index = VectorIndex(dimension=4, default_metric="cosine")
        index.add("a", [1, 0, 0, 0])
        index.update("a", [0, 1, 0, 0])

        results = index.search([0, 1, 0, 0], k=1)
        assert results[0].id == "a"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_max_vectors_enforced(self) -> None:
        index = VectorIndex(dimension=4, max_vectors=2)
        index.add("a", [1, 0, 0, 0])
        index.add("b", [0, 1, 0, 0])
        with pytest.raises(ValueError, match="capacity"):
            index.add("c", [0, 0, 1, 0])


# ---------------------------------------------------------------------------
# Test: distance metrics
# ---------------------------------------------------------------------------


class TestDistanceMetrics:
    def test_cosine_identical_vectors(self) -> None:
        index = VectorIndex(dimension=4, default_metric="cosine")
        index.add("a", [1, 2, 3, 4])
        results = index.search([1, 2, 3, 4], k=1, metric="cosine")
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_euclidean_identical_vectors(self) -> None:
        index = VectorIndex(dimension=4)
        index.add("a", [1, 2, 3, 4])
        results = index.search([1, 2, 3, 4], k=1, metric="euclidean")
        assert results[0].score == pytest.approx(0.0, abs=1e-5)

    def test_dot_product(self) -> None:
        index = VectorIndex(dimension=3)
        index.add("a", [1, 0, 0])
        index.add("b", [0, 1, 0])
        results = index.search([1, 0, 0], k=2, metric="dot_product")
        # "a" should score 1.0, "b" should score 0.0
        assert results[0].id == "a"
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.0)

    def test_euclidean_ordering(self) -> None:
        index = VectorIndex(dimension=2)
        index.add("near", [1, 0])
        index.add("far", [10, 10])
        results = index.search([0, 0], k=2, metric="euclidean")
        assert results[0].id == "near"
        assert results[0].score < results[1].score


# ---------------------------------------------------------------------------
# Test: save / load persistence
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self) -> None:
        index = VectorIndex(dimension=32, default_metric="dot_product")
        for i in range(50):
            index.add(f"doc-{i}", _random_vec(32), metadata={"page": i})

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "test_index"
            index.save(save_path)

            loaded = VectorIndex.load(save_path)

        assert len(loaded) == 50
        assert "doc-0" in loaded
        assert loaded.get_stats()["dimension"] == 32
        assert loaded.get_stats()["default_metric"] == "dot_product"

        # Search results should be identical
        query = _random_vec(32)
        orig_results = index.search(query, k=5, metric="dot_product")
        load_results = loaded.search(query, k=5, metric="dot_product")
        for a, b in zip(orig_results, load_results):
            assert a.id == b.id
            assert a.score == pytest.approx(b.score, abs=1e-5)

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            VectorIndex.load("/nonexistent/path")


# ---------------------------------------------------------------------------
# Test: stats & optimize
# ---------------------------------------------------------------------------


class TestStatsAndOptimize:
    def test_get_stats(self) -> None:
        index = VectorIndex(dimension=16)
        for i in range(10):
            index.add(f"v{i}", _random_vec(16))

        stats = index.get_stats()
        assert stats["num_vectors"] == 10
        assert stats["dimension"] == 16
        assert stats["memory_bytes"] > 0

    def test_optimize_returns_report(self) -> None:
        index = VectorIndex(dimension=8, normalize_on_add=True)
        for i in range(5):
            index.add(f"v{i}", _random_vec(8))

        report = index.optimize()
        assert report["compacted"] is True
        assert report["normalized"] is True
        assert report["num_vectors"] == 5

    def test_from_config(self) -> None:
        cfg = IndexConfig(dimension=64, default_metric="euclidean", max_vectors=500)
        index = VectorIndex.from_config(cfg)
        assert index.get_stats()["dimension"] == 64
        assert index.get_stats()["default_metric"] == "euclidean"

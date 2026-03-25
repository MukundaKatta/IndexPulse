"""Core vector index implementation."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from indexpulse.config import IndexConfig, MetricType
from indexpulse.utils import (
    Vector,
    cosine_similarity_batch,
    dot_product_batch,
    euclidean_distance_batch,
    normalize,
    normalize_matrix,
    validate_id,
    validate_vector,
)

logger = logging.getLogger("indexpulse")

# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class SearchResult:
    """A single search hit returned by :meth:`VectorIndex.search`."""

    score: float
    id: str = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------


class VectorIndex:
    """Lightweight in-memory vector index with monitoring and persistence.

    Parameters
    ----------
    dimension:
        Dimensionality of vectors stored in this index.
    default_metric:
        Default distance metric (``cosine``, ``euclidean``, ``dot_product``).
    normalize_on_add:
        Whether to L2-normalize every vector on :meth:`add`.
    max_vectors:
        Maximum number of vectors allowed (0 = unlimited).
    """

    def __init__(
        self,
        dimension: int = 128,
        default_metric: MetricType = "cosine",
        normalize_on_add: bool = False,
        max_vectors: int = 0,
    ) -> None:
        self._dimension = dimension
        self._default_metric = default_metric
        self._normalize_on_add = normalize_on_add
        self._max_vectors = max_vectors

        # Storage
        self._ids: List[str] = []
        self._id_to_pos: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._vectors: Optional[np.ndarray] = None  # (N, D) float32 matrix
        self._created_at: float = time.time()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: IndexConfig) -> "VectorIndex":
        """Create an index from an :class:`IndexConfig`."""
        return cls(
            dimension=config.dimension,
            default_metric=config.default_metric,
            normalize_on_add=config.normalize_on_add,
            max_vectors=config.max_vectors,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        id: str,
        vector: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a vector into the index.

        Raises
        ------
        ValueError
            If the id already exists or the index is at capacity.
        """
        id = validate_id(id)
        vec = validate_vector(vector, self._dimension)

        if id in self._id_to_pos:
            raise ValueError(f"Vector with id {id!r} already exists — use update()")

        if self._max_vectors and len(self._ids) >= self._max_vectors:
            raise ValueError(
                f"Index is at maximum capacity ({self._max_vectors} vectors)"
            )

        if self._normalize_on_add:
            vec = normalize(vec)

        # Append to storage
        vec_2d = vec.reshape(1, -1)
        if self._vectors is None:
            self._vectors = vec_2d
        else:
            self._vectors = np.vstack([self._vectors, vec_2d])

        pos = len(self._ids)
        self._ids.append(id)
        self._id_to_pos[id] = pos
        self._metadata[id] = metadata or {}
        logger.debug("Added vector %s (pos=%d)", id, pos)

    def delete(self, id: str) -> None:
        """Remove a vector by id.

        Raises
        ------
        KeyError
            If the id does not exist.
        """
        id = validate_id(id)
        if id not in self._id_to_pos:
            raise KeyError(f"Vector {id!r} not found")

        pos = self._id_to_pos.pop(id)
        self._ids.pop(pos)
        self._metadata.pop(id, None)

        assert self._vectors is not None
        self._vectors = np.delete(self._vectors, pos, axis=0)
        if self._vectors.shape[0] == 0:
            self._vectors = None

        # Rebuild position map after deletion
        self._id_to_pos = {vid: i for i, vid in enumerate(self._ids)}
        logger.debug("Deleted vector %s", id)

    def update(
        self,
        id: str,
        vector: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Replace the vector (and optionally metadata) for an existing id.

        Raises
        ------
        KeyError
            If the id does not exist.
        """
        id = validate_id(id)
        if id not in self._id_to_pos:
            raise KeyError(f"Vector {id!r} not found — use add()")

        vec = validate_vector(vector, self._dimension)
        if self._normalize_on_add:
            vec = normalize(vec)

        pos = self._id_to_pos[id]
        assert self._vectors is not None
        self._vectors[pos] = vec

        if metadata is not None:
            self._metadata[id] = metadata
        logger.debug("Updated vector %s (pos=%d)", id, pos)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: Any,
        k: int = 10,
        metric: Optional[MetricType] = None,
    ) -> List[SearchResult]:
        """Find the *k* nearest neighbours to *query_vector*.

        Parameters
        ----------
        query_vector:
            Query vector (same dimensionality as the index).
        k:
            Number of results to return.
        metric:
            Distance metric override (defaults to the index default).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending score for similarity metrics
            and ascending distance for euclidean.
        """
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = validate_vector(query_vector, self._dimension)
        metric = metric or self._default_metric
        k = min(k, len(self._ids))

        scores = self._compute_scores(query, metric)

        if metric == "euclidean":
            # Lower distance = better  ->  take smallest k
            top_indices = np.argpartition(scores, k)[:k]
            top_indices = top_indices[np.argsort(scores[top_indices])]
        else:
            # Higher similarity = better  ->  take largest k
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[SearchResult] = []
        for idx in top_indices:
            vid = self._ids[idx]
            results.append(
                SearchResult(
                    id=vid,
                    score=float(scores[idx]),
                    metadata=self._metadata.get(vid, {}),
                )
            )
        return results

    def _compute_scores(self, query: Vector, metric: MetricType) -> np.ndarray:
        """Return a 1-D array of scores for every stored vector."""
        assert self._vectors is not None
        if metric == "cosine":
            return cosine_similarity_batch(query, self._vectors)
        elif metric == "euclidean":
            return euclidean_distance_batch(query, self._vectors)
        elif metric == "dot_product":
            return dot_product_batch(query, self._vectors)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

    # ------------------------------------------------------------------
    # Stats & monitoring
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return index health and statistics."""
        n = len(self._ids)
        mem_vectors = 0
        if self._vectors is not None:
            mem_vectors = self._vectors.nbytes

        return {
            "num_vectors": n,
            "dimension": self._dimension,
            "default_metric": self._default_metric,
            "normalize_on_add": self._normalize_on_add,
            "max_vectors": self._max_vectors,
            "memory_bytes": mem_vectors,
            "memory_mb": round(mem_vectors / (1024 * 1024), 3),
            "created_at": self._created_at,
            "uptime_seconds": round(time.time() - self._created_at, 2),
        }

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id: str) -> bool:
        return id in self._id_to_pos

    def __repr__(self) -> str:
        return (
            f"VectorIndex(dimension={self._dimension}, "
            f"num_vectors={len(self._ids)}, "
            f"metric={self._default_metric!r})"
        )

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def optimize(self) -> Dict[str, Any]:
        """Compact and normalize index storage.

        Returns a summary dict describing what was done.
        """
        report: Dict[str, Any] = {"normalized": False, "compacted": True}

        if self._vectors is not None:
            # Re-create a contiguous C-order array (removes fragmentation from
            # repeated vstack / delete cycles).
            self._vectors = np.ascontiguousarray(self._vectors, dtype=np.float32)

            if self._normalize_on_add:
                self._vectors = normalize_matrix(self._vectors)
                report["normalized"] = True

        report["num_vectors"] = len(self._ids)
        logger.info("Optimize complete: %s", report)
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the index to disk.

        Creates two files:
        - ``<path>.npz`` — vector matrix
        - ``<path>.index.json`` — ids, metadata, and config

        Returns the base path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save vectors
        npz_path = path.with_suffix(".npz")
        if self._vectors is not None:
            np.savez_compressed(npz_path, vectors=self._vectors)
        else:
            np.savez_compressed(npz_path, vectors=np.empty((0, self._dimension), dtype=np.float32))

        # Save metadata sidecar
        sidecar: Dict[str, Any] = {
            "version": "0.1.0",
            "dimension": self._dimension,
            "default_metric": self._default_metric,
            "normalize_on_add": self._normalize_on_add,
            "max_vectors": self._max_vectors,
            "ids": self._ids,
            "metadata": self._metadata,
        }
        json_path = path.with_suffix(".index.json")
        json_path.write_text(json.dumps(sidecar, indent=2, default=str))

        logger.info("Index saved to %s (%d vectors)", path, len(self._ids))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        """Load a previously saved index from disk."""
        path = Path(path)

        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".index.json")

        if not npz_path.exists():
            raise FileNotFoundError(f"Vector file not found: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Sidecar file not found: {json_path}")

        sidecar = json.loads(json_path.read_text())
        data = np.load(npz_path)
        vectors = data["vectors"]

        index = cls(
            dimension=sidecar["dimension"],
            default_metric=sidecar.get("default_metric", "cosine"),
            normalize_on_add=sidecar.get("normalize_on_add", False),
            max_vectors=sidecar.get("max_vectors", 0),
        )
        index._ids = sidecar["ids"]
        index._id_to_pos = {vid: i for i, vid in enumerate(index._ids)}
        index._metadata = sidecar.get("metadata", {})

        if vectors.shape[0] > 0:
            index._vectors = vectors.astype(np.float32)
        else:
            index._vectors = None

        logger.info("Index loaded from %s (%d vectors)", path, len(index._ids))
        return index

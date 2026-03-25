"""Distance functions, normalization, and validation utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Vector = npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Distance / similarity functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1] where 1 means identical direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    return float(np.linalg.norm(a - b))


def dot_product(a: Vector, b: Vector) -> float:
    """Compute dot product between two vectors."""
    return float(np.dot(a, b))


def cosine_similarity_batch(query: Vector, matrix: Vector) -> Vector:
    """Compute cosine similarity between *query* and every row in *matrix*.

    Returns an array of similarity scores.
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0])
    row_norms = np.linalg.norm(matrix, axis=1)
    # Avoid division by zero for zero-norm stored vectors
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    return matrix @ query / (row_norms * query_norm)


def euclidean_distance_batch(query: Vector, matrix: Vector) -> Vector:
    """Compute Euclidean distance between *query* and every row in *matrix*."""
    return np.linalg.norm(matrix - query, axis=1)


def dot_product_batch(query: Vector, matrix: Vector) -> Vector:
    """Compute dot product between *query* and every row in *matrix*."""
    return matrix @ query


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(vector: Vector) -> Vector:
    """L2-normalize a vector. Returns zero vector if norm is 0."""
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector.copy()
    return vector / norm


def normalize_matrix(matrix: Vector) -> Vector:
    """L2-normalize every row of a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_vector(vector: object, expected_dim: int) -> Vector:
    """Validate and cast *vector* to a numpy float32 array.

    Raises
    ------
    TypeError
        If *vector* is not array-like.
    ValueError
        If dimensionality does not match *expected_dim*.
    """
    if isinstance(vector, np.ndarray):
        arr = vector.astype(np.float32, copy=False)
    else:
        try:
            arr = np.asarray(vector, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"Cannot convert {type(vector).__name__} to a numpy array"
            ) from exc

    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D vector, got shape {arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"Dimension mismatch: expected {expected_dim}, got {arr.shape[0]}"
        )
    return arr


def validate_id(vector_id: str) -> str:
    """Ensure *vector_id* is a non-empty string."""
    if not isinstance(vector_id, str) or not vector_id.strip():
        raise ValueError("vector id must be a non-empty string")
    return vector_id.strip()

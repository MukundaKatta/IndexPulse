"""Configuration models for IndexPulse."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator


MetricType = Literal["cosine", "euclidean", "dot_product"]


class IndexConfig(BaseModel):
    """Configuration for a :class:`VectorIndex`.

    Values can be supplied directly or read from environment variables
    prefixed with ``INDEXPULSE_``.
    """

    dimension: int = Field(
        default_factory=lambda: int(os.getenv("INDEXPULSE_DIMENSION", "128")),
        ge=1,
        description="Dimensionality of vectors stored in the index.",
    )
    default_metric: MetricType = Field(
        default_factory=lambda: os.getenv("INDEXPULSE_DEFAULT_METRIC", "cosine"),  # type: ignore[return-value]
        description="Distance metric used when none is specified at query time.",
    )
    normalize_on_add: bool = Field(
        default_factory=lambda: os.getenv("INDEXPULSE_NORMALIZE_ON_ADD", "false").lower()
        == "true",
        description="Whether to L2-normalize vectors when they are added.",
    )
    max_vectors: int = Field(
        default_factory=lambda: int(os.getenv("INDEXPULSE_MAX_VECTORS", "0")),
        ge=0,
        description="Maximum number of vectors (0 = unlimited).",
    )

    @field_validator("default_metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        allowed = {"cosine", "euclidean", "dot_product"}
        if v not in allowed:
            raise ValueError(f"metric must be one of {allowed}, got {v!r}")
        return v

    model_config = {"frozen": True}

# Architecture

## Overview

IndexPulse is a lightweight, in-memory vector index library designed for embedding into Python applications that need similarity search without running a separate database server.

## Module Structure

```
src/indexpulse/
  __init__.py      Public API surface (VectorIndex, SearchResult, IndexConfig)
  core.py          VectorIndex implementation — CRUD, search, stats, persistence
  config.py        Pydantic configuration model with env-var support
  utils.py         Distance functions, normalization, validation helpers
```

## Key Design Decisions

### Pure NumPy Vectorized Search

All distance computations operate on the full `(N, D)` matrix at once using NumPy broadcasting. This avoids Python-level loops and gives near-C performance for indexes up to ~100k vectors.

For top-k selection the code uses `np.argpartition` (O(N)) rather than a full sort (O(N log N)), which matters at larger index sizes.

### Storage Layout

Vectors are stored in a single contiguous `float32` NumPy array. An ordered list of IDs maps row positions to user-facing identifiers. Metadata is kept in a plain Python dict keyed by ID.

This layout makes batch distance computation trivial and keeps memory overhead low.

### Persistence Format

Saving an index produces two files:

| File | Contents |
|------|----------|
| `<name>.npz` | Compressed NumPy archive containing the vector matrix |
| `<name>.index.json` | JSON sidecar with IDs, metadata, and index configuration |

The split keeps vectors in an efficient binary format while metadata stays human-readable and diffable.

### Configuration

`IndexConfig` is a frozen Pydantic model. Fields can be set directly in code or pulled from `INDEXPULSE_*` environment variables, making it easy to configure via `.env` files in deployment.

## Data Flow

```
add(id, vector, metadata)
  -> validate_vector (dimension check, dtype cast)
  -> optional L2 normalize
  -> np.vstack into _vectors matrix
  -> append id to _ids list
  -> store metadata dict

search(query, k, metric)
  -> validate_vector
  -> batch score computation (cosine / euclidean / dot_product)
  -> np.argpartition for top-k selection
  -> return List[SearchResult]
```

## Limitations & Future Work

- **Linear scan only** — no approximate nearest-neighbour (ANN) structures yet. Works well up to ~100k vectors; beyond that, consider HNSW or IVF partitioning.
- **Single-threaded** — NumPy releases the GIL for most operations, but there is no explicit parallelism.
- **No GPU support** — a future version could swap NumPy for CuPy for GPU-accelerated search.
- **No filtering during search** — metadata filtering happens post-search. Pre-filtering would improve performance for selective queries.

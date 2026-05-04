"""Three-layer news story graph entity directory.

Holds the cache singletons for the full-corpus news story graph built on top
of the ``news_stories`` entity:

* ``bipartite_edges_cache`` - long-form story/attribute-value edges after
  eligibility filtering (FLOOR=2, CAP=5000; excludes ``locations``).
* ``dense_embeddings_cache`` - (N, 1536) L2-normalized text-embedding-3-small
  matrix, with a ``story_ids`` side-car for row alignment.
* ``dense_edges_cache`` - undirected dense similarity edges (``src < dst``) at
  cosine tau = 0.50.
* ``dense_faiss_index_cache`` - trained FAISS IVF index over the dense
  embeddings for fast approximate nearest-neighbor search.
* ``sparse_tfidf_cache`` - (N, 50K) L2-normalized TF-IDF matrix with vocab and
  row-ordered story_id side-cars.
* ``sparse_edges_cache`` - undirected sparse similarity edges (``src < dst``)
  at TF-IDF cosine tau = 0.25.
* ``bm25_vectors_cache`` - (N, K) L2-normalized BM25 sparse matrix produced by
  ``pinecone_text.sparse.BM25Encoder`` with compacted hashed vocabulary.
* ``bm25_edges_cache`` - undirected BM25 similarity edges (``src < dst``) at
  the Q9 winning tau = 0.30.
"""

from __future__ import annotations

from entities.news_stories_graph import (
    bipartite_edges_cache,
    bm25_edges_cache,
    bm25_vectors_cache,
    dense_edges_cache,
    dense_embeddings_cache,
    dense_faiss_index_cache,
    sparse_edges_cache,
    sparse_tfidf_cache,
)

__all__ = [
    "bipartite_edges_cache",
    "bm25_edges_cache",
    "bm25_vectors_cache",
    "dense_edges_cache",
    "dense_embeddings_cache",
    "dense_faiss_index_cache",
    "sparse_edges_cache",
    "sparse_tfidf_cache",
]

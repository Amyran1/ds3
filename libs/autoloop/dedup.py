from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

log = logging.getLogger(__name__)


@dataclass
class DuplicateMatch:
    matched_id: str
    cosine_similarity: float
    matched_title: str


def canonicalize_hypothesis(hypothesis: str) -> str:
    return re.sub(r"\s+", " ", hypothesis).strip().lower()


def content_hash(title: str, row_grain: str, hypothesis: str) -> str:
    normalized = "|".join(
        [
            title.strip().lower(),
            row_grain.strip().lower(),
            canonicalize_hypothesis(hypothesis),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def _embed(container, title: str, hypothesis: str, project: str) -> np.ndarray:
    text = f"{title}\n\n{hypothesis}"
    vecs = await container.openai.embed(
        [text],
        model="text-embedding-3-small",
        cost_key=f"autoloop:{project}:dedup",
    )
    arr = np.asarray(vecs[0], dtype=np.float32)
    # L2-normalize so inner product == cosine similarity
    arr /= np.linalg.norm(arr) + 1e-12
    return arr


class DedupIndex:
    def __init__(self, state_dir: Path, embedding_dim: int = 1536) -> None:
        self._dir = state_dir / "dedup_index"
        self._dim = embedding_dim
        self._metadata: list[dict] = []

        if HAS_FAISS:
            self._faiss_index: faiss.IndexFlatIP | None = None
        else:
            self._array: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)

        self._faiss_path = self._dir / "embeddings.faiss"
        self._npy_path = self._dir / "embeddings.npy"
        self._meta_path = self._dir / "metadata.jsonl"

    def load(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

        if self._meta_path.exists():
            self._metadata = [
                json.loads(line)
                for line in self._meta_path.read_text().splitlines()
                if line.strip()
            ]
        else:
            self._metadata = []

        if HAS_FAISS:
            if self._faiss_path.exists():
                self._faiss_index = faiss.read_index(str(self._faiss_path))
            else:
                self._faiss_index = faiss.IndexFlatIP(self._dim)
        elif self._npy_path.exists():
            self._array = np.load(str(self._npy_path))
        else:
            self._array = np.empty((0, self._dim), dtype=np.float32)

    def save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

        if HAS_FAISS and self._faiss_index is not None:
            tmp = self._faiss_path.with_suffix(".tmp")
            faiss.write_index(self._faiss_index, str(tmp))
            tmp.replace(self._faiss_path)
        else:
            tmp = self._npy_path.with_suffix(".tmp")
            np.save(str(tmp), self._array)
            tmp.replace(self._npy_path)

    async def add(
        self,
        container,
        *,
        item_id: str,
        title: str,
        hypothesis: str,
        project: str,
    ) -> None:
        arr = await _embed(container, title, hypothesis, project)

        vector_index = len(self._metadata)
        meta = {"item_id": item_id, "title": title, "vector_index": vector_index}

        with self._meta_path.open("a") as f:
            f.write(json.dumps(meta) + "\n")
        self._metadata.append(meta)

        if HAS_FAISS:
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(self._dim)
            self._faiss_index.add(np.expand_dims(arr, 0))  # type: ignore[call-arg]  # faiss stubs expose low-level overload; numpy path is correct
        else:
            self._array = np.vstack([self._array, np.expand_dims(arr, 0)])

        self.save()

    async def check(
        self,
        container,
        *,
        title: str,
        hypothesis: str,
        threshold: float,
        project: str,
    ) -> DuplicateMatch | None:
        arr = await _embed(container, title, hypothesis, project)

        n = len(self._metadata)
        if n == 0:
            return None

        if HAS_FAISS and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(np.expand_dims(arr, 0), 1)  # type: ignore[call-arg]  # faiss stubs expose low-level overload; numpy path is correct
            sim = float(distances[0][0])
            row_idx = int(indices[0][0])
        else:
            sims = arr @ self._array.T
            row_idx = int(np.argmax(sims))
            sim = float(sims[row_idx])

        if sim < threshold:
            return None

        meta = self._metadata[row_idx]
        return DuplicateMatch(
            matched_id=meta["item_id"],
            cosine_similarity=sim,
            matched_title=meta["title"],
        )

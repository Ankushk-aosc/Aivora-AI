"""Lightweight in-process vector store + retriever (Part 22).

Deliberately not a distributed system: a numpy matrix and cosine
similarity, which is sufficient for single-document financial Q&A.
"""

from dataclasses import dataclass

import numpy as np

from .chunker import chunk_document
from .document_loader import load_document
from .embeddings import get_embedder


@dataclass
class RetrievedChunk:
    chunk: object
    score: float

    @property
    def citation(self):
        return self.chunk.citation()


class DocumentStore:
    def __init__(self, embedder=None, model=None, device="cpu"):
        self.embedder = embedder or get_embedder(model=model, device=device)
        self.chunks = []
        self.matrix = None
        self.documents = []

    def add_document(self, path, chunk_tokens=256, overlap_tokens=32):
        document = load_document(path)
        chunks = chunk_document(document, chunk_tokens, overlap_tokens)
        if not chunks:
            return {
                "path": path, "chunks_added": 0,
                "note": document.note or "No extractable text found.",
            }
        self.chunks.extend(chunks)
        self.documents.append({"path": path, "format": document.format, "chunks": len(chunks)})
        self._reindex()
        return {"path": path, "format": document.format, "chunks_added": len(chunks)}

    def _reindex(self):
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)

    def search(self, query, top_k=4, min_score=0.0):
        if not self.chunks or self.matrix is None:
            return []
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(self.chunks[i], float(scores[i]))
            for i in order if scores[i] > min_score
        ]

    def stats(self):
        return {
            "documents": self.documents,
            "total_chunks": len(self.chunks),
            "embedder": self.embedder.name,
            "dimensions": None if self.matrix is None else int(self.matrix.shape[1]),
        }


def build_context(retrieved, max_chars=1200):
    """Format retrieved chunks into a context block with real citations."""
    if not retrieved:
        return "", []
    parts, sources, used = [], [], 0
    for r in retrieved:
        text = r.chunk.text.strip()
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        if not text:
            break
        parts.append(text)
        sources.append({"citation": r.citation, "score": round(r.score, 4)})
        used += len(text)
        if used >= max_chars:
            break
    return "\n\n".join(parts), sources

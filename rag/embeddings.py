"""Embeddings for RAG.

Two backends, neither of which calls an external API or loads a
pretrained third-party model:

  ModelEmbedder  - mean-pooled hidden states from THIS repository's
                   DeepSeek-style model (default when a checkpoint is loaded)
  TfidfEmbedder  - dependency-free lexical fallback used when no model is
                   supplied, so RAG remains testable without a checkpoint

Retrieval quality from an undertrained model is poor; TfidfEmbedder is
the honest default for a POC with few training steps.
"""

import math
import re
from collections import Counter

import numpy as np
import torch

from data_sources.tokenizer import get_encoding

_WORD_RE = re.compile(r"\w+")


class TfidfEmbedder:
    """Lexical TF-IDF vectors. Deterministic and model-independent."""

    name = "tfidf"

    def __init__(self):
        self.vocabulary = {}
        self.idf = None

    def _tokenize(self, text):
        return _WORD_RE.findall(text.lower())

    def fit(self, texts):
        df = Counter()
        docs = []
        for text in texts:
            tokens = set(self._tokenize(text))
            docs.append(tokens)
            df.update(tokens)
        self.vocabulary = {term: i for i, term in enumerate(sorted(df))}
        n_docs = max(1, len(texts))
        self.idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for term, i in self.vocabulary.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + df[term])) + 1.0
        return self

    def encode(self, texts):
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        out = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(self._tokenize(text))
            if not counts:
                continue
            total = sum(counts.values())
            for term, count in counts.items():
                col = self.vocabulary.get(term)
                if col is not None:
                    out[row, col] = (count / total) * self.idf[col]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class ModelEmbedder:
    """Mean-pooled hidden states from this repository's own model."""

    name = "deepseek_model"

    def __init__(self, model, device="cpu", max_tokens=256):
        self.model = model
        self.device = device
        self.max_tokens = max_tokens
        self.enc = get_encoding()

    def fit(self, texts):
        return self  # nothing to fit

    @torch.no_grad()
    def _embed_one(self, text):
        ids = self.enc.encode_ordinary(text)[: self.max_tokens]
        if not ids:
            return np.zeros(self.model.config.n_embd, dtype=np.float32)
        idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # Run the model's own stack up to the final norm, then mean-pool.
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=self.device)
        x = self.model.wte(idx) + self.model.wpe(pos)
        for block in self.model.h:
            x = block(x)
        x = self.model.ln_f(x)
        vec = x.mean(dim=1).squeeze(0).float().cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts):
        return np.stack([self._embed_one(t) for t in texts])


def get_embedder(model=None, device="cpu"):
    return ModelEmbedder(model, device=device) if model is not None else TfidfEmbedder()

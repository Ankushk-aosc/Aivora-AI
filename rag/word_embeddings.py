"""A real, locally-trained embedding model (continuous-engineering
priority #3), replacing TF-IDF's purely lexical matching with genuine
dense vectors - trained from scratch on this project's own prepared
corpus (data/shards/), not a downloaded pretrained model, consistent
with the "proprietary, not an external pretrained replacement" rule
that already governs the main LLM.

Skip-gram with negative sampling (word2vec's original formulation),
implemented directly in PyTorch (already a hard dependency - no new
package). Vocabulary is pruned to the most frequent N token IDs from
the corpus; everything else maps to an UNK vector, which keeps the
embedding matrix small enough to train in a few minutes on CPU.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_sources.shard_writer import load_shard_index
from data_sources.tokenizer import get_encoding

MODEL_DIR = os.path.join("checkpoints", "embeddings")
MODEL_PATH = os.path.join(MODEL_DIR, "word_embeddings.pt")
META_PATH = os.path.join(MODEL_DIR, "word_embeddings.json")

UNK_ID = 0


def _load_corpus_ids(shards_root: str = os.path.join("data", "shards"), max_tokens: int = None):
    """Concatenate token IDs from every prepared train shard. Reuses the
    same shard index format the training pipeline already writes -
    no new data format introduced."""
    all_ids = []
    total = 0
    for name in sorted(os.listdir(shards_root)):
        train_dir = os.path.join(shards_root, name, "train")
        index = load_shard_index(train_dir)
        for shard in index.get("shards", []):
            path = os.path.join(train_dir, shard["file"])
            if not os.path.exists(path):
                continue
            arr = np.fromfile(path, dtype=np.uint16)
            all_ids.append(arr)
            total += len(arr)
            if max_tokens and total >= max_tokens:
                break
        if max_tokens and total >= max_tokens:
            break
    if not all_ids:
        raise RuntimeError(f"No prepared shards found under {shards_root}")
    ids = np.concatenate(all_ids)
    return ids[:max_tokens] if max_tokens else ids


def _build_vocab(ids: np.ndarray, vocab_size: int):
    """Keep the `vocab_size` most frequent token IDs; everything else
    maps to UNK_ID=0. Returns (id_to_vocab dict, vocab_to_id array,
    unigram frequency array for negative sampling)."""
    counts = np.bincount(ids, minlength=50257)
    top_ids = np.argsort(-counts)[: vocab_size - 1]  # -1 slot reserved for UNK
    vocab_to_original = np.concatenate([[-1], top_ids])  # index 0 = UNK
    id_to_vocab = {int(orig): i for i, orig in enumerate(vocab_to_original) if orig != -1}
    freqs = np.array([counts[orig] if orig != -1 else 1 for orig in vocab_to_original], dtype=np.float64)
    return id_to_vocab, vocab_to_original, freqs


def _remap_corpus(ids: np.ndarray, id_to_vocab: dict) -> np.ndarray:
    remapped = np.full(len(ids), UNK_ID, dtype=np.int64)
    for orig_id, vocab_id in id_to_vocab.items():
        remapped[ids == orig_id] = vocab_id
    return remapped


class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, center_ids, pos_ids, neg_ids):
        v_c = self.center(center_ids)                      # (B, D)
        v_pos = self.context(pos_ids)                       # (B, D)
        v_neg = self.context(neg_ids)                        # (B, K, D)

        pos_score = torch.sum(v_c * v_pos, dim=1)             # (B,)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)  # (B, K)

        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


def _make_training_pairs(remapped_ids: np.ndarray, window: int, rng: np.random.Generator):
    n = len(remapped_ids)
    centers, contexts = [], []
    for offset in range(1, window + 1):
        centers.append(remapped_ids[offset:])
        contexts.append(remapped_ids[:-offset])
        centers.append(remapped_ids[:-offset])
        contexts.append(remapped_ids[offset:])
    centers = np.concatenate(centers)
    contexts = np.concatenate(contexts)
    keep = (centers != UNK_ID) | (contexts != UNK_ID)
    return centers[keep], contexts[keep]


def train(vocab_size: int = 5000, dim: int = 50, window: int = 4, negatives: int = 5,
          epochs: int = 2, batch_size: int = 2048, max_corpus_tokens: int = 2_000_000,
          seed: int = 42):
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    raw_ids = _load_corpus_ids(max_tokens=max_corpus_tokens)
    id_to_vocab, vocab_to_original, freqs = _build_vocab(raw_ids, vocab_size)
    remapped = _remap_corpus(raw_ids, id_to_vocab)

    centers, contexts = _make_training_pairs(remapped, window, rng)
    n_pairs = len(centers)

    neg_dist = freqs ** 0.75
    neg_dist = neg_dist / neg_dist.sum()

    model = SkipGramNegSampling(vocab_size, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(epochs):
        order = rng.permutation(n_pairs)
        epoch_losses = []
        for start in range(0, n_pairs, batch_size):
            batch_idx = order[start:start + batch_size]
            c = torch.from_numpy(centers[batch_idx]).long()
            p = torch.from_numpy(contexts[batch_idx]).long()
            neg_samples = rng.choice(vocab_size, size=(len(batch_idx), negatives), p=neg_dist)
            n = torch.from_numpy(neg_samples).long()

            optimizer.zero_grad()
            loss = model(c, p, n)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg = float(np.mean(epoch_losses))
        losses.append(avg)
        print(f"epoch {epoch + 1}/{epochs}: avg loss {avg:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "vocab_size": vocab_size, "dim": dim}, MODEL_PATH)

    meta = {
        "vocab_size": vocab_size, "dim": dim, "window": window, "negatives": negatives,
        "epochs": epochs, "corpus_tokens_used": int(len(raw_ids)), "training_pairs": int(n_pairs),
        "seed": seed, "losses_by_epoch": losses,
        "id_to_vocab": {str(k): int(v) for k, v in id_to_vocab.items()},
        "vocab_to_original": vocab_to_original.tolist(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    print(f"Trained in {time.time() - t0:.1f}s on {len(raw_ids):,} corpus tokens, "
          f"{n_pairs:,} training pairs")
    return meta


class TrainedWordEmbedder:
    """Loads the locally-trained skip-gram model and provides
    document/query encoding by averaging word vectors - real dense
    embeddings, not TF-IDF's lexical overlap."""

    name = "trained_skipgram"

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained embedding model at {MODEL_PATH}. Run rag.word_embeddings.train() first."
            )
        saved = torch.load(MODEL_PATH, map_location="cpu")
        with open(META_PATH) as f:
            self.meta = json.load(f)
        self.model = SkipGramNegSampling(saved["vocab_size"], saved["dim"])
        self.model.load_state_dict(saved["model_state_dict"])
        self.model.eval()
        self.id_to_vocab = {int(k): v for k, v in self.meta["id_to_vocab"].items()}
        self.enc = get_encoding()

    def fit(self, texts):
        return self  # nothing to fit - the embedding space is fixed after training

    @torch.no_grad()
    def _embed_one(self, text: str) -> np.ndarray:
        ids = self.enc.encode_ordinary(text)
        vocab_ids = [self.id_to_vocab.get(i, UNK_ID) for i in ids]
        vocab_ids = [v for v in vocab_ids if v != UNK_ID] or [UNK_ID]
        idx = torch.tensor(vocab_ids, dtype=torch.long)
        vectors = self.model.center(idx)
        mean_vec = vectors.mean(dim=0).numpy()
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 0 else mean_vec

    def encode(self, texts):
        return np.stack([self._embed_one(t) for t in texts])

    @torch.no_grad()
    def nearest_neighbors(self, word: str, top_k: int = 8):
        ids = self.enc.encode_ordinary(word)
        if not ids or self.id_to_vocab.get(ids[0], UNK_ID) == UNK_ID:
            return []
        vocab_id = self.id_to_vocab[ids[0]]
        query_vec = F.normalize(self.model.center.weight[vocab_id:vocab_id + 1], dim=1)
        all_vecs = F.normalize(self.model.center.weight, dim=1)
        sims = (all_vecs @ query_vec.T).squeeze(1)
        top = torch.topk(sims, top_k + 1)
        vocab_to_original = self.meta["vocab_to_original"]
        results = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            if idx == vocab_id:
                continue
            original_id = vocab_to_original[idx]
            piece = self.enc.decode([original_id]) if original_id != -1 else "<UNK>"
            results.append((piece, round(score, 4)))
        return results[:top_k]

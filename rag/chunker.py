"""Token-aware chunking that preserves each chunk's real provenance."""

from dataclasses import dataclass

from data_sources.tokenizer import get_encoding


@dataclass
class Chunk:
    text: str
    index: int
    page: int = None
    section: str = None
    source_file: str = ""
    n_tokens: int = 0

    def citation(self) -> str:
        import os
        name = os.path.basename(self.source_file) if self.source_file else "document"
        if self.page is not None:
            return f"{name}, page {self.page}"
        if self.section:
            return f"{name}, {self.section}"
        return name


def chunk_document(document, chunk_tokens: int = 256, overlap_tokens: int = 32):
    """Split a LoadedDocument into token-bounded chunks.

    Page/section metadata is carried from the source section, never
    invented: a chunk inherits only what its originating section reported.
    """
    enc = get_encoding()
    chunks = []
    index = 0

    for section in document.sections:
        ids = enc.encode_ordinary(section.text)
        if not ids:
            continue
        step = max(1, chunk_tokens - overlap_tokens)
        for start in range(0, len(ids), step):
            window = ids[start:start + chunk_tokens]
            if not window:
                continue
            chunks.append(Chunk(
                text=enc.decode(window),
                index=index,
                page=section.page,
                section=section.section,
                source_file=section.source_file or document.path,
                n_tokens=len(window),
            ))
            index += 1
            if start + chunk_tokens >= len(ids):
                break

    return chunks

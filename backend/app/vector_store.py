"""Persistent lightweight vector store for backend-driven RAG."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    source_path: str
    text: str
    metadata: dict[str, str]
    score: float


class VectorStore:
    """Disk-backed bag-of-words vector store with incremental rebuild support."""

    def __init__(self, index_path: str, documents_path: str, chunk_size: int, chunk_overlap: int) -> None:
        self.index_path = Path(index_path)
        self.documents_path = Path(documents_path)
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        self._state: dict[str, object] | None = None

    def is_ready(self) -> bool:
        return self.index_path.exists()

    def load(self) -> None:
        if not self.index_path.exists():
            self._state = None
            return
        self._state = json.loads(self.index_path.read_text(encoding="utf-8"))

    def rebuild(self, chunks: Iterable[DocumentChunk]) -> None:
        chunk_list = list(chunks)
        documents = [self._vectorize(chunk.text) for chunk in chunk_list]
        vocabulary: list[str] = sorted({token for doc in documents for token in doc})
        doc_freq: dict[str, int] = {token: 0 for token in vocabulary}
        for doc in documents:
            for token in set(doc):
                doc_freq[token] += 1

        payload = {
            "vocabulary": vocabulary,
            "document_frequency": doc_freq,
            "documents": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "vector": self._tfidf_vector(doc, vocabulary, doc_freq, len(documents)),
                }
                for chunk, doc in zip(chunk_list, documents)
            ],
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        self._state = payload

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._state is None:
            self.load()
        if self._state is None:
            return []

        vocabulary = list(self._state["vocabulary"])
        doc_freq = dict(self._state["document_frequency"])
        documents = list(self._state["documents"])
        query_tokens = self._vectorize(query)
        query_vector = self._tfidf_vector(query_tokens, vocabulary, doc_freq, len(documents))

        scored: list[ScoredChunk] = []
        for doc in documents:
            score = self._cosine_similarity(query_vector, doc["vector"])
            scored.append(
                ScoredChunk(
                    chunk_id=doc["chunk_id"],
                    source_path=doc["source_path"],
                    text=doc["text"],
                    metadata=dict(doc.get("metadata", {})),
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def build_chunks_from_documents(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for path in sorted(self.documents_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
                continue
            text = self._read_document_text(path)
            for index, chunk in enumerate(self._chunk_text(text)):
                digest = sha256(f"{path.as_posix()}::{index}::{chunk}".encode("utf-8")).hexdigest()
                chunks.append(
                    DocumentChunk(
                        chunk_id=digest,
                        source_path=path.as_posix(),
                        text=chunk,
                        metadata={"filename": path.name, "extension": path.suffix.lower()},
                    )
                )
        return chunks

    def _read_document_text(self, path: Path) -> str:
        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError("pypdf is required for PDF ingestion") from exc
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if path.suffix.lower() == ".docx":
            try:
                from docx import Document as DocxDocument
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError("python-docx is required for DOCX ingestion") from exc
            document = DocxDocument(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)

        return ""

    def _chunk_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        words = normalized.split(" ")
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self.chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _vectorize(self, text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"\b\w+\b", text)]

    def _tfidf_vector(self, tokens: list[str], vocabulary: list[str], doc_freq: dict[str, int], doc_count: int) -> list[float]:
        counts = Counter(tokens)
        length = len(tokens) or 1
        vector: list[float] = []
        for term in vocabulary:
            tf = counts.get(term, 0) / length
            idf = math.log((1 + doc_count) / (1 + doc_freq.get(term, 0))) + 1.0
            vector.append(tf * idf)
        return vector

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

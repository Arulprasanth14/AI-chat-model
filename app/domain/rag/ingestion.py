"""
app/domain/rag/ingestion.py
────────────────────────────
Knowledge document ingestion pipeline.

Reads markdown documents from a profile's knowledge_docs/ folder,
chunks them semantically, embeds each chunk, and upserts to the vector store.

Chunking strategy: paragraph-aware split targeting ~400 tokens per chunk
with ~50 token overlap. Metadata (doc_type, field_code) is inferred from
document frontmatter and filename conventions.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import tiktoken

from app.domain.rag.embedder import Embedder
from app.domain.rag.vector_store import ChunkToUpsert, VectorStore

logger = logging.getLogger(__name__)

# Tokenizer for chunk size estimation (cl100k_base ≈ gpt-4 tokenizer)
# Note: google/embeddinggemma-300m has an 8192-token context window, so these
# chunk sizes will never cause silent truncation (unlike all-mpnet-base-v2).
_tokenizer = tiktoken.get_encoding("cl100k_base")

# Target and maximum chunk sizes in tokens.
# embeddinggemma-300m supports 8192 tokens — using 512/600 for high-quality
# coherent chunks while keeping prompts manageable.
TARGET_CHUNK_TOKENS = 512
MAX_CHUNK_TOKENS = 600
OVERLAP_TOKENS = 64

# Embedding batch size — Gemma is efficient; 32 at once saves inference overhead
EMBED_BATCH_SIZE = 32


def _count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base tokenizer."""
    return len(_tokenizer.encode(text))


def _extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-style frontmatter from a markdown document.

    Frontmatter is expected between ``---`` delimiters at the top of the file.
    Falls back to empty dict if no frontmatter is found.

    Returns:
        (metadata dict, remaining document body)
    """
    import yaml

    fm_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = fm_pattern.match(content)
    if match:
        try:
            meta = yaml.safe_load(match.group(1)) or {}
            body = content[match.end():]
            return meta, body
        except Exception:
            pass
    return {}, content


def _infer_doc_type_from_filename(filename: str) -> str:
    """Infer doc_type tag from filename if not in frontmatter.

    Conventions:
      question_guidance.md  → "question_guidance"
      domain_facts.md       → "domain_fact"
      examples.md           → "example"
    """
    stem = Path(filename).stem.lower()
    if "question" in stem or "guidance" in stem:
        return "question_guidance"
    if "fact" in stem or "domain" in stem:
        return "domain_fact"
    if "example" in stem:
        return "example"
    return "general"


def _paragraph_aware_chunk(text: str) -> list[str]:
    """Split text into chunks using paragraph boundaries.

    Strategy:
    1. Split on double-newlines (paragraph boundaries).
    2. Accumulate paragraphs until TARGET_CHUNK_TOKENS is reached.
    3. If a single paragraph exceeds MAX_CHUNK_TOKENS, split on sentences.
    4. Add OVERLAP_TOKENS of context from the previous chunk.

    Returns:
        List of text chunks.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    overlap_tail = ""

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # Single paragraph too large → sentence-split it
        if para_tokens > MAX_CHUNK_TOKENS:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                s_tokens = _count_tokens(sentence)
                if current_tokens + s_tokens > TARGET_CHUNK_TOKENS and current_parts:
                    chunk_text = (overlap_tail + " " if overlap_tail else "") + " ".join(current_parts)
                    chunks.append(chunk_text.strip())
                    # Compute overlap tail from last ~OVERLAP_TOKENS tokens
                    overlap_tail = _get_tail_tokens(chunk_text, OVERLAP_TOKENS)
                    current_parts = []
                    current_tokens = 0
                current_parts.append(sentence)
                current_tokens += s_tokens
            continue

        if current_tokens + para_tokens > TARGET_CHUNK_TOKENS and current_parts:
            chunk_text = (overlap_tail + " " if overlap_tail else "") + "\n\n".join(current_parts)
            chunks.append(chunk_text.strip())
            overlap_tail = _get_tail_tokens(chunk_text, OVERLAP_TOKENS)
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        chunk_text = (overlap_tail + " " if overlap_tail else "") + "\n\n".join(current_parts)
        chunks.append(chunk_text.strip())

    return [c for c in chunks if c.strip()]


def _get_tail_tokens(text: str, n_tokens: int) -> str:
    """Return the last n_tokens worth of text (for overlap)."""
    tokens = _tokenizer.encode(text)
    tail_tokens = tokens[-n_tokens:]
    return _tokenizer.decode(tail_tokens)


def _stable_chunk_id(profile_id: str, source_file: str, chunk_index: int) -> str:
    """Generate a stable deterministic UUID for a chunk.

    Stable IDs enable idempotent upserts — re-ingesting the same document
    updates existing chunks rather than creating duplicates.
    """
    key = f"{profile_id}::{source_file}::{chunk_index}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


class KnowledgeIngester:
    """Ingests knowledge documents into the vector store.

    Args:
        embedder:     Embedder for vectorising text chunks.
        vector_store: VectorStore backend for persistence.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest_profile(
        self,
        profile_id: str,
        knowledge_docs_path: str | Path,
        clear_existing: bool = True,
    ) -> dict[str, int]:
        """Ingest all documents in a profile's knowledge_docs/ folder.

        Args:
            profile_id:          Profile namespace for this ingestion run.
            knowledge_docs_path: Path to the knowledge_docs/ directory.
            clear_existing:      If True, delete all existing chunks before ingesting.

        Returns:
            Dict with keys: docs_processed, chunks_created, chunks_upserted.
        """
        docs_path = Path(knowledge_docs_path)
        if not docs_path.exists():
            raise FileNotFoundError(f"knowledge_docs path not found: {docs_path}")

        if clear_existing:
            deleted = await self._vector_store.delete_by_profile(profile_id)
            logger.info("Cleared existing chunks", extra={"profile_id": profile_id, "deleted": deleted})

        md_files = list(docs_path.rglob("*.md"))
        if not md_files:
            logger.warning("No .md files found", extra={"path": str(docs_path)})
            return {"docs_processed": 0, "chunks_created": 0, "chunks_upserted": 0}

        all_chunks: list[ChunkToUpsert] = []

        for doc_path in md_files:
            chunks = self._process_document(doc_path, profile_id)
            all_chunks.extend(chunks)
            logger.info(
                "Document processed",
                extra={"file": doc_path.name, "chunks": len(chunks)},
            )

        # Embed in batches
        upserted = await self._embed_and_upsert(all_chunks)

        stats = {
            "docs_processed": len(md_files),
            "chunks_created": len(all_chunks),
            "chunks_upserted": upserted,
        }
        logger.info("Ingestion complete", extra=stats)
        return stats

    def _process_document(
        self, doc_path: Path, profile_id: str
    ) -> list[ChunkToUpsert]:
        """Read, parse, chunk, and prepare a single document for embedding."""
        content = doc_path.read_text(encoding="utf-8")
        frontmatter, body = _extract_frontmatter(content)

        doc_type = frontmatter.get("doc_type") or _infer_doc_type_from_filename(doc_path.name)
        industry = frontmatter.get("industry")
        brief_type = frontmatter.get("brief_type")
        field_code = frontmatter.get("field_code")

        text_chunks = _paragraph_aware_chunk(body)
        chunks: list[ChunkToUpsert] = []

        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = _stable_chunk_id(profile_id, doc_path.name, idx)
            chunks.append(
                ChunkToUpsert(
                    chunk_id=chunk_id,
                    profile_id=profile_id,
                    content=chunk_text,
                    embedding=[],  # filled during embed step
                    doc_type=doc_type,
                    source_file=doc_path.name,
                    chunk_index=idx,
                    industry=industry,
                    brief_type=brief_type,
                    field_code=field_code,
                )
            )

        return chunks

    async def _embed_and_upsert(self, chunks: list[ChunkToUpsert]) -> int:
        """Embed chunks in batches and upsert to the vector store."""
        total_upserted = 0

        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]

            embeddings = await self._embedder.embed_batch(texts)

            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

            upserted = await self._vector_store.upsert(batch)
            total_upserted += upserted

            logger.debug(
                "Batch upserted",
                extra={"batch_start": batch_start, "batch_size": len(batch)},
            )

        return total_upserted

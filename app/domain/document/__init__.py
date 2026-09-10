"""
app/domain/document/__init__.py
────────────────────────────────
Document extraction domain package.

Exports the primary public surface:
    DocumentExtractor      — orchestrates LLM field extraction from a ParsedDocument
    DocumentExtractionResult — result dataclass returned by the extractor
"""
from app.domain.document.document_extractor import (
    DocumentExtractor,
    DocumentExtractionResult,
)

__all__ = ["DocumentExtractor", "DocumentExtractionResult"]

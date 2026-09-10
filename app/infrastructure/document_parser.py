"""
app/infrastructure/document_parser.py
──────────────────────────────────────
Multi-format document parser for the document-upload field-extraction feature.

Converts uploaded file bytes into a ParsedDocument (raw text + semantic chunks)
that the DocumentExtractor can feed to the LLM for field extraction.

Supported formats:
    .pdf   — pdfplumber (text + table extraction per page)
    .docx  — python-docx (paragraphs, tables, headings)
    .txt   — UTF-8 / Latin-1 decode
    .csv   — stdlib csv (converts rows to "key: value" lines)
    .md    — treated as plain text (markdown stripped superficially)

All other formats raise DocumentParseError.
Image files (image/*) should be routed to the /logo endpoint — not here.

SECURITY NOTES:
    - pdfplumber and python-docx do NOT execute embedded scripts or macros.
    - No document bytes are persisted — parsing happens entirely in memory.
    - Null bytes are stripped before returning to prevent PostgreSQL JSONB errors.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Approximate characters per token (conservative estimate for chunking budget).
_CHARS_PER_TOKEN: int = 4

# Max characters per semantic chunk (~2,000 tokens when divided by _CHARS_PER_TOKEN).
_MAX_CHUNK_CHARS: int = 8_000

# Minimum chunk size — don't create tiny orphan chunks.
_MIN_CHUNK_CHARS: int = 200

# Supported MIME types for document (non-image) processing.
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({
    "text/plain",
    "text/csv",
    "text/markdown",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc (will attempt docx path; may fail)
    "application/octet-stream",  # generic binary — route by extension
})

# Extension → parser mapping (used when MIME type is ambiguous/generic).
_EXT_TO_PARSER: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "text",
    ".text": "text",
    ".md": "text",
    ".markdown": "text",
    ".csv": "csv",
    ".tsv": "csv",
}


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single semantic chunk of document text.

    Attributes:
        text:        The chunk's content (stripped, null-byte-free).
        chunk_index: Zero-based position within the document.
        page_number: Source page (1-based) if available; None for non-paged docs.
        section:     Section heading context if available (e.g. "Introduction").
        chunk_type:  One of "text", "table", "list", "heading".
    """
    text: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    chunk_type: str = "text"  # "text" | "table" | "list" | "heading"


@dataclass
class ParsedDocument:
    """Result of a successful document parse.

    Attributes:
        raw_text:     Full extracted text (all pages/sections concatenated).
        chunks:       Semantic chunks ready for LLM extraction.
        page_count:   Number of pages (1 for non-paged formats).
        parser_used:  Which parser produced this result (for logging).
        warnings:     Non-fatal issues encountered during parsing.
        filename:     Original filename for traceability logging.
    """
    raw_text: str
    chunks: list[DocumentChunk]
    page_count: int
    parser_used: str
    warnings: list[str] = field(default_factory=list)
    filename: str = ""


class DocumentParseError(Exception):
    """Raised when a document cannot be parsed."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message: str = user_message or message


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_document(
    content: bytes,
    filename: str,
    content_type: str | None = None,
    max_chunk_chars: int = _MAX_CHUNK_CHARS,
) -> ParsedDocument:
    """Parse uploaded document bytes into a structured ParsedDocument.

    Routing logic:
      1. Determine parser from MIME type or file extension.
      2. Parse raw text from file bytes.
      3. Sanitize (strip null bytes, normalize whitespace).
      4. Split into semantic chunks.

    Args:
        content:        Raw file bytes from the upload.
        filename:       Original filename (used for extension-based routing).
        content_type:   MIME type from the upload (may be None or generic).
        max_chunk_chars: Maximum characters per chunk (default: 8,000 chars ≈ 2k tokens).

    Returns:
        ParsedDocument with chunks ready for extraction.

    Raises:
        DocumentParseError: On unsupported format, corrupted file, or empty text.
    """
    parser_name = _resolve_parser(filename, content_type)

    logger.info(
        "Document parser selected",
        extra={"file_name": filename, "content_type": content_type, "parser": parser_name},
    )

    if parser_name == "pdf":
        raw_text, page_count, warnings, chunks_from_parser = _parse_pdf(content, filename)
    elif parser_name == "docx":
        raw_text, page_count, warnings, chunks_from_parser = _parse_docx(content, filename)
    elif parser_name == "text":
        raw_text, page_count, warnings, chunks_from_parser = _parse_text(content, filename)
    elif parser_name == "csv":
        raw_text, page_count, warnings, chunks_from_parser = _parse_csv(content, filename)
    else:
        raise DocumentParseError(
            f"No parser for format: {parser_name!r}",
            user_message=(
                f"Sorry, '{filename}' is not a supported document type. "
                "Please upload a PDF, Word document (.docx), plain text (.txt), or CSV file."
            ),
        )

    # Sanitize raw text
    raw_text = _sanitize(raw_text)

    if not raw_text.strip():
        raise DocumentParseError(
            f"Document produced no extractable text: {filename!r}",
            user_message=(
                f"'{filename}' appears to be empty or contains no readable text. "
                "If it's a scanned PDF or image-only document, please upload a text-based version."
            ),
        )

    # If the parser already produced structured chunks (e.g. DOCX), use those.
    # Otherwise, split the raw text into chunks.
    if chunks_from_parser:
        chunks = chunks_from_parser
        # Still apply max_chunk_chars to any oversized chunks
        chunks = _split_oversized_chunks(chunks, max_chunk_chars)
    else:
        chunks = _split_into_chunks(raw_text, max_chunk_chars)

    logger.info(
        "Document parsed successfully",
        extra={
            "file_name": filename,
            "parser": parser_name,
            "char_count": len(raw_text),
            "chunk_count": len(chunks),
            "page_count": page_count,
        },
    )

    return ParsedDocument(
        raw_text=raw_text,
        chunks=chunks,
        page_count=page_count,
        parser_used=parser_name,
        warnings=warnings,
        filename=filename,
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate for a text string (4 chars ≈ 1 token)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ── Parser implementations ─────────────────────────────────────────────────────

def _parse_pdf(
    content: bytes,
    filename: str,
) -> tuple[str, int, list[str], list[DocumentChunk]]:
    """Parse a PDF using pdfplumber.

    Extracts text per page and converts tables to readable key:value rows.
    Returns (raw_text, page_count, warnings, chunks).
    """
    try:
        import pdfplumber  # lazy import — only required for PDF uploads
    except ImportError as exc:
        raise DocumentParseError(
            "pdfplumber is not installed",
            user_message="PDF parsing is not available. Please install pdfplumber or upload a .txt file.",
        ) from exc

    warnings: list[str] = []
    all_chunks: list[DocumentChunk] = []
    raw_parts: list[str] = []
    chunk_index = 0

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            page_count = len(pdf.pages)

            if page_count == 0:
                raise DocumentParseError(
                    f"PDF has 0 pages: {filename!r}",
                    user_message=f"'{filename}' appears to be an empty or corrupted PDF.",
                )

            for page_num, page in enumerate(pdf.pages, start=1):
                page_parts: list[str] = []

                # Extract plain text
                page_text = page.extract_text() or ""
                page_text = _sanitize(page_text)
                if page_text.strip():
                    page_parts.append(page_text)

                # Extract tables separately — convert to readable text blocks
                tables = page.extract_tables() or []
                for table in tables:
                    table_lines: list[str] = []
                    for row in table:
                        if row:
                            cells = [str(c).strip() if c is not None else "" for c in row]
                            table_lines.append(" | ".join(cells))
                    if table_lines:
                        table_text = "[TABLE]\n" + "\n".join(table_lines) + "\n[/TABLE]"
                        page_parts.append(table_text)

                if not page_parts:
                    warnings.append(f"Page {page_num} produced no extractable text.")
                    continue

                page_full_text = "\n\n".join(page_parts)
                raw_parts.append(page_full_text)

                # One chunk per page (will be split further if oversized)
                all_chunks.append(DocumentChunk(
                    text=page_full_text,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    chunk_type="text",
                ))
                chunk_index += 1

    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"PDF parse failed for {filename!r}: {exc}",
            user_message=(
                f"Could not read '{filename}'. The file may be corrupted or password-protected. "
                "Please try saving it as a plain .txt file and uploading again."
            ),
        ) from exc

    if not raw_parts:
        raise DocumentParseError(
            f"PDF contains no extractable text: {filename!r}",
            user_message=(
                f"'{filename}' appears to be a scanned or image-based PDF with no readable text. "
                "Please upload a text-based PDF or a .txt version of the document."
            ),
        )

    return "\n\n".join(raw_parts), page_count, warnings, all_chunks


def _parse_docx(
    content: bytes,
    filename: str,
) -> tuple[str, int, list[str], list[DocumentChunk]]:
    """Parse a DOCX file using python-docx.

    Extracts paragraphs and tables in document order, preserving heading hierarchy.
    Returns (raw_text, page_count, warnings, chunks).
    """
    try:
        from docx import Document as DocxDocument  # lazy import
    except ImportError as exc:
        raise DocumentParseError(
            "python-docx is not installed",
            user_message="DOCX parsing is not available. Please upload a .pdf or .txt file instead.",
        ) from exc

    warnings: list[str] = []
    all_chunks: list[DocumentChunk] = []
    raw_parts: list[str] = []
    chunk_index = 0
    current_section: str | None = None
    current_section_lines: list[str] = []

    def _flush_section() -> None:
        nonlocal chunk_index
        if not current_section_lines:
            return
        text = "\n".join(current_section_lines).strip()
        if text:
            raw_parts.append(text)
            all_chunks.append(DocumentChunk(
                text=text,
                chunk_index=chunk_index,
                section=current_section,
                chunk_type="text",
            ))
            chunk_index += 1
        current_section_lines.clear()

    try:
        doc = DocxDocument(io.BytesIO(content))

        for block in _iter_docx_blocks(doc):
            block_type = block.get("type", "text")
            block_text = _sanitize(block.get("text", ""))

            if not block_text.strip():
                continue

            if block_type == "heading":
                # Flush previous section, start a new one
                _flush_section()
                current_section = block_text
                # Add the heading itself as the first line of the new section
                current_section_lines.append(f"## {block_text}")

            elif block_type == "table":
                # Flush any accumulated prose before the table
                _flush_section()
                table_text = f"[TABLE]\n{block_text}\n[/TABLE]"
                raw_parts.append(table_text)
                all_chunks.append(DocumentChunk(
                    text=table_text,
                    chunk_index=chunk_index,
                    section=current_section,
                    chunk_type="table",
                ))
                chunk_index += 1

            else:
                current_section_lines.append(block_text)

        _flush_section()

    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"DOCX parse failed for {filename!r}: {exc}",
            user_message=(
                f"Could not read '{filename}'. The file may be corrupted or in an old .doc format. "
                "Please save it as .docx or export it as a .txt file and upload again."
            ),
        ) from exc

    if not raw_parts:
        raise DocumentParseError(
            f"DOCX contains no extractable text: {filename!r}",
            user_message=f"'{filename}' appears to be empty or contains only images.",
        )

    return "\n\n".join(raw_parts), 1, warnings, all_chunks


def _iter_docx_blocks(doc: object) -> list[dict]:
    """Iterate over DOCX document body yielding typed blocks.

    Returns dicts with keys: type (heading|paragraph|table|list_item), text.
    """
    from docx.oxml.ns import qn  # type: ignore[import]
    import docx  # type: ignore[import]

    blocks: list[dict] = []

    body = doc.element.body  # type: ignore[attr-defined]
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            para = docx.text.paragraph.Paragraph(child, doc)  # type: ignore[attr-defined]
            style_name = (para.style.name or "").lower() if para.style else ""
            text = para.text.strip()
            if not text:
                continue
            if style_name.startswith("heading"):
                blocks.append({"type": "heading", "text": text})
            elif style_name.startswith("list"):
                blocks.append({"type": "list_item", "text": f"• {text}"})
            else:
                blocks.append({"type": "paragraph", "text": text})

        elif tag == "tbl":
            tbl = docx.table.Table(child, doc)  # type: ignore[attr-defined]
            rows: list[str] = []
            for row in tbl.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                blocks.append({"type": "table", "text": "\n".join(rows)})

    return blocks


def _parse_text(
    content: bytes,
    filename: str,
) -> tuple[str, int, list[str], list[DocumentChunk]]:
    """Parse a plain text or Markdown file.

    Returns (raw_text, page_count=1, warnings, chunks=[]).
    Chunking is left to the caller (_split_into_chunks).
    """
    warnings: list[str] = []

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
            warnings.append("File was not valid UTF-8; decoded as Latin-1.")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                f"Could not decode {filename!r}: {exc}",
                user_message=(
                    f"'{filename}' contains characters that could not be decoded. "
                    "Please save it as UTF-8 and upload again."
                ),
            ) from exc

    # Superficially strip Markdown syntax for cleaner extraction
    if filename.lower().endswith((".md", ".markdown")):
        text = _strip_markdown(text)

    return _sanitize(text), 1, warnings, []


def _parse_csv(
    content: bytes,
    filename: str,
) -> tuple[str, int, list[str], list[DocumentChunk]]:
    """Parse a CSV/TSV file into human-readable key:value text blocks.

    If the CSV has a header row, each data row becomes "Header: Value" lines.
    Without a header, rows become pipe-separated strings.
    Returns (raw_text, page_count=1, warnings, chunks=[]).
    """
    warnings: list[str] = []

    try:
        text_raw = content.decode("utf-8")
    except UnicodeDecodeError:
        text_raw = content.decode("latin-1")
        warnings.append("CSV was not valid UTF-8; decoded as Latin-1.")

    delimiter = "\t" if filename.lower().endswith(".tsv") else ","

    try:
        reader = csv.reader(io.StringIO(text_raw), delimiter=delimiter)
        rows = list(reader)
    except csv.Error as exc:
        raise DocumentParseError(
            f"CSV parse error in {filename!r}: {exc}",
            user_message=f"Could not parse '{filename}' as a CSV file. Please check its format.",
        ) from exc

    if not rows:
        raise DocumentParseError(
            f"Empty CSV: {filename!r}",
            user_message=f"'{filename}' appears to be empty.",
        )

    # Attempt header detection (first row treated as header)
    header = rows[0]
    data_rows = rows[1:]

    lines: list[str] = []
    if data_rows and all(h.strip() for h in header):
        for row in data_rows:
            pairs = []
            for i, cell in enumerate(row):
                col_name = header[i] if i < len(header) else f"Column {i+1}"
                if cell.strip():
                    pairs.append(f"{col_name}: {cell.strip()}")
            if pairs:
                lines.append("\n".join(pairs))
    else:
        for row in rows:
            lines.append(" | ".join(c.strip() for c in row if c.strip()))

    return _sanitize("\n\n".join(lines)), 1, warnings, []


# ── Chunking helpers ───────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_chars: int) -> list[DocumentChunk]:
    """Split text into semantic chunks, preferring paragraph boundaries.

    Algorithm:
      1. Split on double-newlines (paragraph breaks).
      2. Accumulate paragraphs into chunks until the chunk would exceed max_chars.
      3. If a single paragraph exceeds max_chars, split on sentence boundaries.

    Returns:
        List of DocumentChunk with chunk_type="text".
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[DocumentChunk] = []
    current_lines: list[str] = []
    current_chars = 0
    chunk_index = 0

    for para in paragraphs:
        if len(para) > max_chars:
            # Oversized paragraph: split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_chars + len(sentence) > max_chars and current_lines:
                    chunks.append(DocumentChunk(
                        text="\n\n".join(current_lines),
                        chunk_index=chunk_index,
                        chunk_type="text",
                    ))
                    chunk_index += 1
                    current_lines = []
                    current_chars = 0
                current_lines.append(sentence)
                current_chars += len(sentence)
        else:
            if current_chars + len(para) > max_chars and current_lines:
                chunks.append(DocumentChunk(
                    text="\n\n".join(current_lines),
                    chunk_index=chunk_index,
                    chunk_type="text",
                ))
                chunk_index += 1
                current_lines = []
                current_chars = 0
            current_lines.append(para)
            current_chars += len(para)

    if current_lines:
        chunks.append(DocumentChunk(
            text="\n\n".join(current_lines),
            chunk_index=chunk_index,
            chunk_type="text",
        ))

    return chunks or [DocumentChunk(text=text[:max_chars], chunk_index=0, chunk_type="text")]


def _split_oversized_chunks(
    chunks: list[DocumentChunk],
    max_chars: int,
) -> list[DocumentChunk]:
    """Further split any chunk that exceeds max_chars.

    Preserves all metadata (page_number, section, chunk_type) on sub-chunks.
    """
    result: list[DocumentChunk] = []
    new_index = 0

    for chunk in chunks:
        if len(chunk.text) <= max_chars:
            result.append(DocumentChunk(
                text=chunk.text,
                chunk_index=new_index,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
            ))
            new_index += 1
        else:
            sub_chunks = _split_into_chunks(chunk.text, max_chars)
            for sub in sub_chunks:
                result.append(DocumentChunk(
                    text=sub.text,
                    chunk_index=new_index,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    chunk_type=chunk.chunk_type,
                ))
                new_index += 1

    return result


# ── Sanitization helpers ───────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Strip null bytes and normalize whitespace."""
    # Strip null bytes — prevents PostgreSQL JSONB crashes
    text = text.replace("\x00", "")
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _strip_markdown(text: str) -> str:
    """Superficially strip common Markdown formatting for cleaner LLM extraction."""
    # Remove ATX headings markers (keep text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove links — keep display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


# ── Format resolution ──────────────────────────────────────────────────────────

def _resolve_parser(filename: str, content_type: str | None) -> str:
    """Determine which parser to use for the uploaded file.

    Priority:
      1. File extension (most reliable signal).
      2. MIME type (fallback for extensionless files).

    Raises DocumentParseError for unsupported formats.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in _EXT_TO_PARSER:
        return _EXT_TO_PARSER[ext]

    # Fallback: use MIME type
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if "pdf" in mime:
            return "pdf"
        if "word" in mime or "docx" in mime or "document" in mime:
            return "docx"
        if "csv" in mime or "comma" in mime:
            return "csv"
        if "text" in mime:
            return "text"

    raise DocumentParseError(
        f"Unsupported document format: ext={ext!r} mime={content_type!r}",
        user_message=(
            f"'{filename}' is not a supported document type. "
            "Please upload a PDF (.pdf), Word document (.docx), "
            "plain text (.txt), or CSV (.csv) file."
        ),
    )

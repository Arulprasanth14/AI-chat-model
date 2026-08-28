"""Change embedding dimension from 1536 to 768 on knowledge_chunks.embedding

Revision ID: fa6751537373
Revises:
Create Date: 2026-08-26 11:33:36.890309

Plain-English summary of EVERY operation this migration performs
═══════════════════════════════════════════════════════════════════
upgrade():
  1. DROP INDEX  knowledge_chunks_embedding_idx
     (the IVFFlat cosine index on the old 1536-dim column — must go first
      because the column type change would make it invalid anyway)
  2. ALTER COLUMN  knowledge_chunks.embedding
     TYPE vector(768) USING embedding::vector(768)
     (changes the pgvector dimension from 1536 → 768; existing row data is
      cast, but vectors stored at 1536-dim will be truncated/invalid — a
      full re-embed via scripts/reembed_all.py is required after running this)
  3. CREATE INDEX  knowledge_chunks_embedding_idx
     USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)
     (rebuilds the cosine-similarity index for the new 768-dim column)

downgrade():
  1. DROP INDEX  knowledge_chunks_embedding_idx  (768-dim version)
  2. ALTER COLUMN  knowledge_chunks.embedding
     TYPE vector(1536) USING embedding::vector(1536)
     (reverts to 1536-dim; data loss on existing rows is acceptable per spec)
  3. CREATE INDEX  knowledge_chunks_embedding_idx
     USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)
     (rebuilds the index for the 1536-dim column)

NO other tables are created, altered, or dropped.
═══════════════════════════════════════════════════════════════════

⚠️  SIMILARITY THRESHOLD NOTE (manual re-tuning required after running this):
    The RAGRetriever uses a cosine similarity threshold that was tuned for
    text-embedding-3-small (1536-dim).  After switching to
    google/embedding-gemma-3-300m-it-v0 (768-dim) you MUST re-tune:

      • config value:   VECTOR_SIMILARITY_THRESHOLD  (currently 0.7 in .env)
      • code location:  app/core/config.py  → vector_similarity_threshold
      • injected into:  app/api/deps.py     → get_retriever()
      • used at:        app/domain/rag/retriever.py → RAGRetriever.__init__
                        (line ~52, filtered at line ~104)

    Do NOT silently change the threshold — re-tune with live retrieval evals.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = 'fa6751537373'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade: change knowledge_chunks.embedding from vector(1536) to vector(768).

    Steps:
      1. Drop the IVFFlat index (type change makes it invalid).
      2. Alter the column type to vector(768).
      3. Rebuild the IVFFlat index for the new dimension.
    """
    # ── 1. Drop the existing IVFFlat index ────────────────────────────────────
    op.drop_index(
        'knowledge_chunks_embedding_idx',
        table_name='knowledge_chunks',
        postgresql_using='ivfflat',
    )

    # ── 2. Alter column: vector(1536) → vector(768) ───────────────────────────
    # The USING clause casts existing data; any stored 1536-dim vectors will
    # be truncated to 768 dims — run scripts/reembed_all.py afterward.
    op.execute(
        sa.text(
            "ALTER TABLE knowledge_chunks "
            "ALTER COLUMN embedding TYPE vector(768) "
            "USING embedding::vector(768)"
        )
    )

    # ── 3. Rebuild the cosine-similarity index for 768-dim ────────────────────
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE tablename = 'knowledge_chunks'
                    AND indexname = 'knowledge_chunks_embedding_idx'
                ) THEN
                    CREATE INDEX knowledge_chunks_embedding_idx
                    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                END IF;
            END
            $$;
            """
        )
    )


def downgrade() -> None:
    """Downgrade: revert knowledge_chunks.embedding from vector(768) to vector(1536).

    Data loss on existing rows is acceptable (per spec) — the vectors are
    simply re-cast (truncated) back to 1536 dims.  Run scripts/reembed_all.py
    with EMBEDDING_PROVIDER=openai to repopulate after reverting.
    """
    # ── 1. Drop the 768-dim IVFFlat index ─────────────────────────────────────
    op.drop_index(
        'knowledge_chunks_embedding_idx',
        table_name='knowledge_chunks',
        postgresql_using='ivfflat',
    )

    # ── 2. Revert column: vector(768) → vector(1536) ──────────────────────────
    op.execute(
        sa.text(
            "ALTER TABLE knowledge_chunks "
            "ALTER COLUMN embedding TYPE vector(1536) "
            "USING embedding::vector(1536)"
        )
    )

    # ── 3. Rebuild the 1536-dim cosine index ──────────────────────────────────
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE tablename = 'knowledge_chunks'
                    AND indexname = 'knowledge_chunks_embedding_idx'
                ) THEN
                    CREATE INDEX knowledge_chunks_embedding_idx
                    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                END IF;
            END
            $$;
            """
        )
    )

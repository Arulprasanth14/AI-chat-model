"""
app/domain/llm/openai_provider.py
───────────────────────────────────
Backward-compatibility re-export shim.

The canonical OpenAI LLM provider implementation has been moved to the
infrastructure layer where it belongs:

    app/infrastructure/llm/openai_provider.py

This shim exists so that existing scripts, diagnostic tools, and tests that
import from this path continue to work without modification:

    from app.domain.llm.openai_provider import OpenAIProvider  # still works

New code should import directly from the infrastructure layer:

    from app.infrastructure.llm.openai_provider import OpenAIProvider
"""
from app.infrastructure.llm.openai_provider import OpenAIProvider

__all__ = ["OpenAIProvider"]

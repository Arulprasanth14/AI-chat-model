"""
Manual E2E transcript — verifies Bug 4 fix (guidance-source log now visible).
Run with: python e2e_transcript.py
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path

# Enable DEBUG so the guidance-source log line fires
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s  %(name)s: %(message)s",
)
# Silence noisy third-party libs
for _lib in ("httpcore", "httpx", "openai", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

from dotenv import load_dotenv

load_dotenv()

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.structural_resolver import StructuralResolver
from app.domain.llm.openai_provider import OpenAIProvider
from app.domain.llm.prompt_builder import PromptBuilder
from app.domain.rag.retriever import RAGRetriever
from app.domain.rag.vector_store import VectorSearchResult
from app.infrastructure.persistence.session_repository import InMemorySessionRepository
from app.project_profiles.base_profile import BaseProfile
from tests.evaluation.conftest import ConfigurableMockVectorStore, MockEmbedder


async def run_turn(orch: ConversationOrchestrator, sid: str | None, user_msg: str) -> dict:
    print(f"\nUSER: {user_msg}")
    assistant_text = ""
    final_snap: dict = {}
    actual_sid = sid
    async for event in orch.process_turn(session_id=sid, user_message=user_msg):
        raw = event.replace("data: ", "").strip()
        try:
            obj = json.loads(raw)
            if "snapshot" in obj:
                final_snap = obj["snapshot"]
                actual_sid = final_snap["session_id"]
        except Exception:
            pass
    if not actual_sid:
        print("Error: No session ID available. The turn likely failed before a snapshot could be saved.")
        return final_snap
        
    session = await orch._repo.get_session(actual_sid)
    if not session:
        print(f"Error: Session '{actual_sid}' not found.")
        return final_snap
        
    for turn in reversed(session.conversation_history):
        if turn["role"] == "assistant":
            assistant_text = turn["content"]
            break
    print(f"ASSISTANT: {assistant_text.encode('ascii', 'ignore').decode('ascii')}")
    if final_snap:
        captured = list(final_snap.get("extracted_answers", {}).keys())
        missing = [mf["field_code"] for mf in final_snap.get("missing_fields", [])]
        print(f"  Captured fields : {captured}")
        print(f"  Missing fields  : {missing}")
        print(f"  is_complete     : {final_snap['is_complete']}")
        print(f"  Session ID      : {final_snap['session_id'][:8]}...")
    return final_snap


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "")

    llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
    embedder = MockEmbedder()
    store = ConfigurableMockVectorStore(
        [
            VectorSearchResult(
                chunk_id="fallback",
                content=(
                    "Ask the client about company name, project type and budget "
                    "to complete a creative brief."
                ),
                doc_type="question_guidance",
                score=0.9,
            )
        ]
    )
    repo = InMemorySessionRepository()
    retriever = RAGRetriever(embedder=embedder, vector_store=store)
    builder = PromptBuilder()

    resolver = StructuralResolver(
        field_sets_root=Path("app/project_profiles/picasso_fusion/field_sets"),
        knowledge_docs_root=Path("app/project_profiles/picasso_fusion/knowledge_docs"),
    )
    prof = BaseProfile.from_yaml(Path("app/project_profiles/picasso_fusion/profile.yaml"))

    from app.api.deps import _make_profile_provider
    orch = ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=builder,
        profile_provider=_make_profile_provider(prof),
        structural_resolver=resolver,
    )

    print("=" * 60)
    print("TURN 1 — on-script (restaurant static post)")
    print("=" * 60)
    snap1 = await run_turn(orch, None, "Hi! I need to create a restaurant static post.")
    sid = snap1["session_id"]

    print()
    print("=" * 60)
    print("TURN 2 — off-script (joke request)")
    print("=" * 60)
    await run_turn(orch, sid, "Can you tell me a joke instead?")


if __name__ == "__main__":
    asyncio.run(main())

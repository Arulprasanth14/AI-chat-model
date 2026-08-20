# run_live_benchmark.py
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.structural_resolver import StructuralResolver
from app.domain.llm.openai_provider import OpenAIProvider
from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from app.domain.rag.retriever import RAGRetriever
from app.domain.rag.vector_store import VectorSearchResult
from app.infrastructure.persistence.session_repository import InMemorySessionRepository
from app.project_profiles.base_profile import BaseProfile
from tests.evaluation.conftest import ConfigurableMockVectorStore, MockEmbedder
from tests.evaluation.conftest import evaluate_answer_relevance, evaluate_faithfulness

test_cases = [
    {
        "user_msg": "My company is Acme Corp and our budget is $15,000 for a static post.",
        "expected_topic": "project_type", 
        "chunks": [
            VectorSearchResult("chunk1", "Standard static post packages start at $5,000.", "knowledge", 0.9)
        ]
    },
    {
        "user_msg": "We want 100% ROI guarantee.",
        "expected_topic": "target_audience",
        "chunks": [
            VectorSearchResult("chunk2", "We never guarantee specific ROI. We focus on engagement.", "knowledge", 0.9)
        ]
    },
    {
        "user_msg": "Can you design a website for $500?",
        "expected_topic": "deliverables",
        "chunks": [
            VectorSearchResult("chunk3", "Website design packages start at $10,000 minimum.", "knowledge", 0.9)
        ]
    },
    {
        "user_msg": "I am looking for a beauty wellness campaign.",
        "expected_topic": "primary_objective",
        "chunks": [
            VectorSearchResult("chunk4", "Our beauty campaigns are known for minimal aesthetics.", "knowledge", 0.9)
        ]
    },
    {
        "user_msg": "We don't know our target audience.",
        "expected_topic": "target_audience",
        "chunks": [
            VectorSearchResult("chunk5", "If client doesn't know audience, suggest our strategy workshop.", "knowledge", 0.9)
        ]
    },
]

async def run_benchmark():
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIProvider(api_key=api_key, model="gpt-4o")
    embedder = MockEmbedder()
    prof = BaseProfile.from_yaml(Path("app/project_profiles/picasso_fusion/profile.yaml"))
    
    total_faith = 0
    total_rel = 0
    evaluator_llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini") # fast evaluator
    
    for i, tc in enumerate(test_cases):
        store = ConfigurableMockVectorStore(tc["chunks"])
        repo = InMemorySessionRepository()
        retriever = RAGRetriever(embedder=embedder, vector_store=store)
        builder = PromptBuilder()
        resolver = StructuralResolver(Path("app/project_profiles/picasso_fusion/field_sets"), Path("app/project_profiles/picasso_fusion/knowledge_docs"))
        
        orch = ConversationOrchestrator(
            session_repo=repo,
            retriever=retriever,
            llm=llm,
            prompt_builder=builder,
            profile_provider=lambda _s: prof,
            structural_resolver=resolver,
        )
        
        assistant_text = ""
        import json
        async for event in orch.process_turn(session_id=None, user_message=tc["user_msg"]):
            raw = event.replace("data: ", "").strip()
            try:
                obj = json.loads(raw)
                if "snapshot" in obj:
                    sid = obj["snapshot"]["session_id"]
            except Exception:
                pass
        
        session = await repo.get_session(sid)
        for turn in reversed(session.conversation_history):
            if turn["role"] == "assistant":
                assistant_text = turn["content"]
                break
        
        print(f"\n--- Case {i+1} ---")
        print(f"USER: {tc['user_msg']}")
        print(f"ASSISTANT: {assistant_text}")
        
        # Evaluate
        rel_score = await evaluate_answer_relevance(evaluator_llm, tc["user_msg"], assistant_text, expected_next_topic=tc["expected_topic"])
        
        chunks = [c.content for c in tc["chunks"]]
        faith_score = await evaluate_faithfulness(evaluator_llm, assistant_text, chunks)
        
        print(f"Relevance: {rel_score:.2f} | Faithfulness: {faith_score:.2f}")
        total_rel += rel_score
        total_faith += faith_score
        
    print("\n=== RESULTS ===")
    print(f"Average Answer Relevance: {total_rel/len(test_cases):.2f}")
    print(f"Average Faithfulness: {total_faith/len(test_cases):.2f}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())

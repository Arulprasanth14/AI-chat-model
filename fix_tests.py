import os
import glob

# Fix RAGRetriever
for fpath in glob.glob("tests/evaluation/metrics/*.py"):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    if "vector_store=store, top_k=" in content:
        content = content.replace("vector_store=store, top_k=", "vector_store=store, similarity_threshold=0.0, top_k=")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

# Fix test_state_transition.py intent
with open("tests/evaluation/metrics/test_state_transition.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('"model_believes_complete": False,\n    }', '"model_believes_complete": False,\n        "intent": "ask_question",\n    }')
content = content.replace('"model_believes_complete": True,\n    }', '"model_believes_complete": True,\n        "intent": "ask_question",\n    }')

with open("tests/evaluation/metrics/test_state_transition.py", "w", encoding="utf-8") as f:
    f.write(content)

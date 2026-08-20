"""
tests/evaluation/run_evaluation.py
────────────────────────────────────
Evaluation suite runner and reporter.

Usage:
    python tests/evaluation/run_evaluation.py
    python tests/evaluation/run_evaluation.py --no-live   # skip live API tests

This script:
  1. Discovers and runs all pytest tests in tests/evaluation/metrics/.
  2. Collects metric scores emitted into the EVAL_METRICS dict by each test.
  3. Prints a formatted summary table matching the spec in the implementation plan.
  4. Returns exit code 0 if all tests pass, 1 if any fail.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Picasso AI evaluation suite and print metric results."
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip tests marked @pytest.mark.live (no OpenAI API calls).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save metric results as JSON.",
    )
    return parser.parse_args()


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_pytest(skip_live: bool = False) -> tuple[int, float]:
    """Run pytest on the evaluation suite and capture output."""
    project_root = Path(__file__).parent.parent.parent
    eval_dir = Path(__file__).parent / "metrics"

    cmd = [
        sys.executable, "-m", "pytest",
        str(eval_dir),
        "-v",
        "--tb=short",
        "-p", "no:warnings",
        "--no-header",
        "-s",  # allow print() output for metric reporting
    ]

    if skip_live:
        cmd += ["-m", "not live"]

    print(f"\n{'='*70}")
    print("  PICASSO AI EVALUATION SUITE")
    print(f"{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Mode: {'Deterministic only (--no-live)' if skip_live else 'Full (including live API tests)'}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(project_root),
    )
    elapsed = time.time() - start

    return result.returncode, elapsed


# ── Report ─────────────────────────────────────────────────────────────────────

def safe_avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "N/A (no data)"
    if pct:
        return f"{value:.1%}"
    return f"{value:.4f}"


def print_report(metrics: dict, elapsed: float) -> None:
    """Print the formatted evaluation report."""

    cosine = safe_avg(metrics.get("cosine_similarity", []))
    precision = safe_avg(metrics.get("context_precision", []))
    recall = safe_avg(metrics.get("context_recall", []))
    ext_prec = safe_avg(metrics.get("extraction_precision", []))
    ext_recall = safe_avg(metrics.get("extraction_recall", []))
    faithfulness = safe_avg(metrics.get("faithfulness", []))
    ans_rel = safe_avg(metrics.get("answer_relevance", []))
    state_correct = safe_avg(metrics.get("state_transition_correct", []))
    turns = metrics.get("turns_to_completion", [])
    avg_turns = safe_avg(turns)
    goal_success = safe_avg(metrics.get("goal_completion_success", []))

    w = 56
    print(f"\n{'='*w}")
    print("  EVALUATION REPORT")
    print(f"{'='*w}")

    print("\n  RAG METRICS")
    print(f"  {'Average Cosine Similarity':<35} {fmt(cosine)}")
    print(f"  {'Context Precision':<35} {fmt(precision, pct=True)}")
    print(f"  {'Context Recall':<35} {fmt(recall, pct=True)}")

    print("\n  LLM METRICS")
    print(f"  {'Extraction Precision':<35} {fmt(ext_prec, pct=True)}")
    print(f"  {'Extraction Recall':<35} {fmt(ext_recall, pct=True)}")
    print(f"  {'Faithfulness':<35} {fmt(faithfulness, pct=True)}")
    print(f"  {'Answer Relevance':<35} {fmt(ans_rel, pct=True)}")

    print("\n  END-TO-END METRICS")
    print(f"  {'State Transition Correctness':<35} {fmt(state_correct, pct=True)}")
    print(f"  {'Goal Completion Rate':<35} {fmt(goal_success, pct=True)}")
    print(f"  {'Average Turns to Completion':<35} {f'{avg_turns:.1f}' if avg_turns else 'N/A'}")
    if turns:
        print(f"  {'  Min/Max Turns':<35} {min(turns):.0f} / {max(turns):.0f}")

    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"{'='*w}")


def collect_metrics_from_conftest() -> dict:
    """Import and read the EVAL_METRICS dict from conftest after tests run.

    Note: pytest runs in a subprocess, so EVAL_METRICS will not be populated
    in this process. We use a JSON report via a conftest pytest hook instead.
    This function reads that file if it exists.
    """
    metrics_path = Path(__file__).parent / "_metrics_report.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            return json.load(f)
    return {}


# ── Conftest hook injector ─────────────────────────────────────────────────────
# We add a hook to conftest.py at runtime to serialize EVAL_METRICS to JSON.
# This is how the runner reads metric scores after the subprocess completes.

HOOK_CODE = '''

# ── Auto-generated by run_evaluation.py ───────────────────────────────────────
import json as _json
from pathlib import Path as _Path

def pytest_sessionfinish(session, exitstatus):
    """Serialize EVAL_METRICS to JSON so run_evaluation.py can read the scores."""
    output_path = _Path(__file__).parent / "_metrics_report.json"
    try:
        output_path.write_text(_json.dumps(EVAL_METRICS, indent=2))
    except Exception:
        pass
'''


def ensure_hook_injected() -> None:
    """Ensure the pytest_sessionfinish hook is in conftest.py."""
    conftest_path = Path(__file__).parent / "conftest.py"
    content = conftest_path.read_text(encoding="utf-8")
    if "pytest_sessionfinish" not in content:
        conftest_path.write_text(content + HOOK_CODE, encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    ensure_hook_injected()

    returncode, elapsed = run_pytest(skip_live=args.no_live)

    # Read metrics that were serialized by the pytest hook
    metrics = collect_metrics_from_conftest()

    print_report(metrics, elapsed)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved to: {args.output_json}")

    if returncode != 0:
        print(f"\n  [!] Some tests failed (exit code: {returncode}). See output above.")
    else:
        print(f"\n  [OK] All tests passed.")

    return returncode


if __name__ == "__main__":
    sys.exit(main())

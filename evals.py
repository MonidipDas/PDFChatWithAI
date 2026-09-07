"""
Enhanced LLM Evaluation Framework for PDFChatWithAI.

Evaluates the full RAG pipeline (retrieval + generation) across multiple
dimensions using both deterministic keyword checks and an LLM-as-judge
approach. Results are printed as a formatted table and saved to
eval_results.json.

Usage:
    python evals.py
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from pdfchat.config import configure_api_key, validate_api_key, make_api_request
from pdfchat.embeddings import create_vector_store
from pdfchat.qa import get_answer

# ---------------------------------------------------------------------------
# Sample documents – simulating text extracted from a short PDF
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT_TEXT = """
The Solar System consists of the Sun and the objects that orbit it.
The largest of these are the eight planets, which form two main groups:
the inner terrestrial planets (Mercury, Venus, Earth, and Mars), and the
outer giant planets (Jupiter, Saturn, Uranus, and Neptune).
Jupiter is the largest planet in the solar system, and Mercury is the
smallest.  The asteroid belt lies between the orbits of Mars and Jupiter.

Earth is the third planet from the Sun and the only known planet to
harbour life.  It has one natural satellite, the Moon.  Earth's
atmosphere is composed primarily of nitrogen (78%) and oxygen (21%).

Saturn is famous for its prominent ring system, which is composed mainly
of ice particles with a smaller amount of rocky debris and dust.
Neptune is the farthest known planet from the Sun in the Solar System.
Pluto was reclassified as a dwarf planet in 2006 by the International
Astronomical Union (IAU).
"""

# ---------------------------------------------------------------------------
# Test dataset – 8 diverse questions covering different difficulty levels
# ---------------------------------------------------------------------------

TEST_QUESTIONS: List[Dict[str, Any]] = [
    # --- Factual recall ---
    {
        "question": "Which is the largest planet in the solar system?",
        "expected_facts": ["Jupiter"],
        "category": "factual_recall",
    },
    {
        "question": "Which is the smallest planet in the solar system?",
        "expected_facts": ["Mercury"],
        "category": "factual_recall",
    },
    # --- Location / spatial reasoning ---
    {
        "question": "Where is the asteroid belt located?",
        "expected_facts": ["Mars", "Jupiter"],
        "category": "spatial_reasoning",
    },
    # --- Composition / detail ---
    {
        "question": "What is Earth's atmosphere primarily composed of?",
        "expected_facts": ["nitrogen", "oxygen"],
        "category": "detail_extraction",
    },
    # --- Multi-hop reasoning ---
    {
        "question": "What are the inner terrestrial planets and the outer giant planets?",
        "expected_facts": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
        "category": "multi_hop",
    },
    # --- Specific fact ---
    {
        "question": "What is Saturn's ring system composed of?",
        "expected_facts": ["ice"],
        "category": "detail_extraction",
    },
    # --- Classification / reclassification ---
    {
        "question": "Why is Pluto no longer considered a planet?",
        "expected_facts": ["dwarf planet", "2006"],
        "category": "reasoning",
    },
    # --- Hallucination resistance (unanswerable from context) ---
    {
        "question": "What is the surface temperature of Pluto?",
        "expected_facts": [],  # no facts in context – model should say it doesn't know
        "category": "hallucination_resistance",
    },
]

# ---------------------------------------------------------------------------
# Deterministic keyword-based fact verification
# ---------------------------------------------------------------------------


def keyword_fact_check(answer: str, expected_facts: List[str]) -> Dict[str, Any]:
    """Check whether each expected fact appears in the answer (case-insensitive).

    Returns a dict with:
        - matched: list of matched keywords
        - missed: list of missed keywords
        - score: fraction of matched keywords (0.0–1.0)
        - is_unanswerable: True if no facts are expected (hallucination test)
    """
    if not expected_facts:
        # For unanswerable questions, we *want* the model to NOT fabricate.
        # Heuristic: if the answer contains hedging language, that's good.
        hedging_phrases = [
            "not mentioned", "no information", "not available",
            "don't have", "doesn't mention", "not provided",
            "cannot determine", "not specified", "i don't know",
            "not in the context", "not enough information",
            "unable to", "cannot answer", "no data",
        ]
        answer_lower = answer.lower()
        hedged = any(phrase in answer_lower for phrase in hedging_phrases)
        return {
            "matched": [],
            "missed": [],
            "score": 1.0 if hedged else 0.0,
            "is_unanswerable": True,
            "hedging_detected": hedged,
        }

    answer_lower = answer.lower()
    matched = [f for f in expected_facts if f.lower() in answer_lower]
    missed = [f for f in expected_facts if f.lower() not in answer_lower]
    score = len(matched) / len(expected_facts) if expected_facts else 0.0

    return {
        "matched": matched,
        "missed": missed,
        "score": round(score, 2),
        "is_unanswerable": False,
    }


# ---------------------------------------------------------------------------
# LLM-as-Judge  (6 scored dimensions)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG)
system.  Evaluate the following ANSWER based on the QUESTION and the
CONTEXT that was retrieved from a document store.

Return your evaluation as a JSON object with EXACTLY these keys (no
extra text outside the JSON):

{{
  "context_relevance":  <1-5>,
  "answer_correctness": <1-5>,
  "faithfulness":       <1-5>,
  "completeness":       <1-5>,
  "conciseness":        <1-5>,
  "reasoning":          "<one-sentence explanation>"
}}

Scoring guide (1 = worst, 5 = best):
  context_relevance  – How relevant is the retrieved CONTEXT to the QUESTION?
  answer_correctness – Is the ANSWER factually correct given the CONTEXT?
  faithfulness       – Does the ANSWER stick to the CONTEXT without hallucinating?
  completeness       – Does the ANSWER cover all aspects of the QUESTION?
  conciseness        – Is the ANSWER appropriately concise, not verbose?

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}

JSON:
"""

METRIC_KEYS = [
    "context_relevance",
    "answer_correctness",
    "faithfulness",
    "completeness",
    "conciseness",
]


def llm_judge(question: str, context: str, answer: str) -> Dict[str, Any]:
    """Use the Groq API as an LLM judge to score the response on 5 dimensions.

    Returns a dict with metric scores (1–5) and a reasoning string.
    On failure, returns default scores of 0 for every metric.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        answer=answer,
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    default_result = {k: 0 for k in METRIC_KEYS}
    default_result["reasoning"] = "LLM judge call failed"

    try:
        response = make_api_request(
            "POST", "/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )
        parsed = json.loads(content)

        # Validate that all expected keys are present and in range
        result: Dict[str, Any] = {}
        for key in METRIC_KEYS:
            val = parsed.get(key, 0)
            result[key] = max(1, min(5, int(val))) if val else 0
        result["reasoning"] = parsed.get("reasoning", "")
        return result

    except Exception as exc:
        print(f"  [WARN] LLM judge error: {exc}")
        return default_result


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all evaluation results."""
    total = len(results)
    if total == 0:
        return {}

    # Average LLM-judge metrics
    avg_metrics: Dict[str, float] = {}
    for key in METRIC_KEYS:
        scores = [r["llm_judge"].get(key, 0) for r in results if r["llm_judge"].get(key, 0) > 0]
        avg_metrics[key] = round(sum(scores) / len(scores), 2) if scores else 0.0

    # Average keyword fact score
    fact_scores = [r["keyword_check"]["score"] for r in results]
    avg_fact_score = round(sum(fact_scores) / len(fact_scores), 2)

    # Average latency
    latencies = [r["latency_seconds"] for r in results]
    avg_latency = round(sum(latencies) / len(latencies), 2)

    # Pass rates (score >= 4 is a "pass")
    pass_rates: Dict[str, float] = {}
    for key in METRIC_KEYS:
        passing = [r for r in results if r["llm_judge"].get(key, 0) >= 4]
        pass_rates[key] = round(len(passing) / total * 100, 1)

    return {
        "total_questions": total,
        "avg_metrics": avg_metrics,
        "avg_keyword_fact_score": avg_fact_score,
        "avg_latency_seconds": avg_latency,
        "pass_rates_pct": pass_rates,
    }


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------


def print_header(text: str) -> None:
    """Print a formatted section header."""
    width = 72
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_question_result(idx: int, result: Dict[str, Any]) -> None:
    """Print a single question's evaluation result."""
    print(f"\n{'─' * 72}")
    print(f"  Q{idx + 1} [{result['category']}]: {result['question']}")
    print(f"{'─' * 72}")
    print(f"  Answer : {result['answer'][:120]}{'...' if len(result['answer']) > 120 else ''}")
    print(f"  Latency: {result['latency_seconds']:.2f}s")

    # Keyword check
    kw = result["keyword_check"]
    if kw.get("is_unanswerable"):
        status = "[PASS] Hedging detected" if kw.get("hedging_detected") else "[FAIL] No hedging (possible hallucination)"
        print(f"  Keywords: N/A (unanswerable) -> {status}")
    else:
        print(f"  Keywords: {kw['score']:.0%}  matched={kw['matched']}  missed={kw['missed']}")

    # LLM judge
    judge = result["llm_judge"]
    if judge.get("reasoning") == "LLM judge call failed":
        print("  LLM Judge: [WARN] Failed (using keyword score only)")
    else:
        scores_str = "  ".join(f"{k[:6]}={judge[k]}/5" for k in METRIC_KEYS)
        print(f"  LLM Judge: {scores_str}")
        if judge.get("reasoning"):
            print(f"  Reasoning: {judge['reasoning']}")


def print_summary_table(agg: Dict[str, Any]) -> None:
    """Print a formatted summary table of aggregate metrics."""
    print_header("EVALUATION SUMMARY")

    print(f"\n  Total Questions Evaluated : {agg['total_questions']}")
    print(f"  Avg Keyword Fact Score   : {agg['avg_keyword_fact_score']:.0%}")
    print(f"  Avg Latency              : {agg['avg_latency_seconds']:.2f}s")

    print(f"\n  {'Metric':<25} {'Avg Score':>10} {'Pass Rate':>10}")
    print(f"  {'─' * 45}")
    for key in METRIC_KEYS:
        avg = agg["avg_metrics"].get(key, 0)
        pr = agg["pass_rates_pct"].get(key, 0)
        label = key.replace("_", " ").title()
        print(f"  {label:<25} {avg:>8.2f}/5 {pr:>8.1f}%")

    # Overall quality score (average of all LLM-judge metrics)
    all_avgs = [v for v in agg["avg_metrics"].values() if v > 0]
    overall = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else 0
    print(f"\n  {'─' * 45}")
    print(f"  {'OVERALL QUALITY':<25} {overall:>8.2f}/5")
    print()


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------


def save_results(results: List[Dict[str, Any]], agg: Dict[str, Any], filepath: str = "eval_results.json") -> None:
    """Save detailed results and aggregated summary to a JSON file."""
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": agg,
        "per_question": results,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Results saved to {filepath}")


# ---------------------------------------------------------------------------
# Core evaluation function (importable by Streamlit dashboard)
# ---------------------------------------------------------------------------


def evaluate_single_question(
    test: Dict[str, Any],
    retriever,
) -> Dict[str, Any]:
    """Evaluate a single test question and return the result dict.

    This is the atomic evaluation unit, usable from both CLI and Streamlit.
    """
    question = test["question"]
    expected_facts = test["expected_facts"]
    category = test["category"]

    # --- Retrieve + Answer (measure latency) ---
    start_time = time.perf_counter()

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = get_answer(question, retriever)

    latency = round(time.perf_counter() - start_time, 2)

    # --- Keyword fact check (deterministic) ---
    kw_result = keyword_fact_check(answer, expected_facts)

    # --- LLM-as-Judge (5 scored metrics) ---
    judge_result = llm_judge(question, context, answer)

    return {
        "question": question,
        "category": category,
        "expected_facts": expected_facts,
        "answer": answer,
        "context_snippet": context[:300],
        "latency_seconds": latency,
        "keyword_check": kw_result,
        "llm_judge": judge_result,
    }


def evaluate_live_query(
    question: str,
    answer: str,
    context: str,
    latency_seconds: float,
) -> Dict[str, Any]:
    """Evaluate a real user query using the LLM judge.

    Unlike evaluate_single_question(), this does NOT need expected_facts
    or a predefined test dict.  It is designed for live monitoring of
    actual user interactions.

    Args:
        question: The user's question.
        answer: The generated answer.
        context: The retrieved context used to generate the answer.
        latency_seconds: Time taken for retrieval + generation.

    Returns:
        A result dict with llm_judge scores and metadata.
    """
    judge_result = llm_judge(question, context, answer)

    return {
        "question": question,
        "category": "user_query",
        "expected_facts": [],
        "answer": answer,
        "context_snippet": context[:300],
        "latency_seconds": latency_seconds,
        "keyword_check": {"score": 0, "matched": [], "missed": [], "is_unanswerable": False},
        "llm_judge": judge_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_evaluation(
    questions: List[Dict[str, Any]] | None = None,
    document_text: str | None = None,
) -> tuple:
    """Run the full evaluation pipeline and return (results, aggregate).

    Args:
        questions: List of test question dicts. Defaults to TEST_QUESTIONS.
        document_text: Source text for the retriever. Defaults to SAMPLE_DOCUMENT_TEXT.

    Returns:
        Tuple of (per_question_results, aggregate_summary).
    """
    questions = questions or TEST_QUESTIONS
    document_text = document_text or SAMPLE_DOCUMENT_TEXT

    configure_api_key()
    validate_api_key()

    retriever = create_vector_store(document_text)

    results: List[Dict[str, Any]] = []
    for test in questions:
        result = evaluate_single_question(test, retriever)
        results.append(result)

    agg = aggregate_results(results)
    return results, agg


# ---------------------------------------------------------------------------
# CLI runner (console output + JSON save)
# ---------------------------------------------------------------------------


def run_evals() -> None:
    """Run the full evaluation pipeline with formatted console output."""

    print_header("PDFChatWithAI -- LLM Evaluation Framework")

    # 1. Configure API
    print("\n  [*] Configuring API key...")
    configure_api_key()
    validate_api_key()
    print("  [OK] API key validated")

    # 2. Build retriever from sample document
    print("\n  [*] Building hybrid retriever + cross-encoder reranker...")
    retriever = create_vector_store(SAMPLE_DOCUMENT_TEXT)
    print("  [OK] Retriever ready")

    # 3. Evaluate each question
    results: List[Dict[str, Any]] = []

    print_header(f"EVALUATING {len(TEST_QUESTIONS)} QUESTIONS")

    for idx, test in enumerate(TEST_QUESTIONS):
        result = evaluate_single_question(test, retriever)
        results.append(result)
        print_question_result(idx, result)

    # 4. Aggregate
    agg = aggregate_results(results)

    # 5. Print summary
    print_summary_table(agg)

    # 6. Save to JSON
    save_results(results, agg)

    print("  [OK] Evaluation complete!\n")


if __name__ == "__main__":
    run_evals()


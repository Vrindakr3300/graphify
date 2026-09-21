from __future__ import annotations
import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from .gemini_client import get_gemini_service
from .baseline import run_baseline_query
from .pipeline.orchestrator import run_graphrank_pipeline

def _grade_key_facts_heuristically(answer: str, key_facts: List[str]) -> tuple[float, List[str]]:
    """Deterministic, reproducible key fact grading using normalized token coverage."""
    ans_lower = answer.lower()
    covered_facts = []
    score_sum = 0.0

    for fact in key_facts:
        fact_tokens = [t.lower() for t in fact.split() if len(t) > 2]
        if not fact_tokens:
            continue
        matches = sum(1 for tok in fact_tokens if tok in ans_lower)
        ratio = matches / len(fact_tokens)

        if ratio >= 0.75:
            score_sum += 1.0
            covered_facts.append(f"Full: '{fact}'")
        elif ratio >= 0.40:
            score_sum += 0.5
            covered_facts.append(f"Partial: '{fact}'")

    total = max(1, len(key_facts))
    coverage = round(score_sum / total, 3)
    return coverage, covered_facts

async def grade_with_gemini_judge(
    question: str,
    answer: str,
    key_facts: List[str]
) -> tuple[float, List[str]]:
    """Grading using Gemini as an independent judge, with automatic fallback to heuristic."""
    gemini = get_gemini_service()
    if not gemini.is_configured:
        return _grade_key_facts_heuristically(answer, key_facts)

    prompt = f"""You are a strict, objective grading judge evaluating code QA accuracy.
Question: "{question}"

Gold Key Facts required for a complete answer:
{json.dumps(key_facts, indent=2)}

Candidate Answer:
\"\"\"{answer}\"\"\"

For each key fact, decide if the answer covers it:
- "covered": fact is clearly and accurately stated
- "partial": fact is mentioned vaguely or incomplete
- "missing": fact is absent or contradicted

Respond ONLY with valid JSON:
{{
  "fact_ratings": [
    {{"fact": "...", "verdict": "covered|partial|missing", "quote": "verbatim quote or none"}}
  ]
}}
"""
    try:
        raw = await gemini.generate_text(prompt, json_mode=True, temperature=0.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        ratings = data.get("fact_ratings", [])
        
        covered_count = 0.0
        details = []
        for r in ratings:
            v = r.get("verdict", "missing").lower()
            f = r.get("fact", "")
            if v == "covered":
                covered_count += 1.0
                details.append(f"Full: '{f}'")
            elif v == "partial":
                covered_count += 0.5
                details.append(f"Partial: '{f}'")
                
        total = max(1, len(key_facts))
        return round(covered_count / total, 3), details
    except Exception:
        return _grade_key_facts_heuristically(answer, key_facts)

async def evaluate_single_question(item: Dict[str, Any]) -> Dict[str, Any]:
    qid = item["id"]
    tier = item.get("tier", "medium")
    question = item["question"]
    key_facts = item.get("key_facts", [])

    # Run Baseline
    b_ans, b_cands = await run_baseline_query(question, top_n=10)
    b_score, b_details = await grade_with_gemini_judge(question, b_ans.answer, key_facts)

    # Run GraphRank
    g_ans, trace = await run_graphrank_pipeline(question, top_candidates=8, allow_multihop=True)
    g_score, g_details = await grade_with_gemini_judge(question, g_ans.answer, key_facts)

    return {
        "id": qid,
        "tier": tier,
        "question": question,
        "key_facts_count": len(key_facts),
        "baseline": {
            "score": b_score,
            "covered": b_details,
            "tokens": b_ans.token_usage_approx,
            "answer_preview": b_ans.answer[:200]
        },
        "graphrank": {
            "score": g_score,
            "covered": g_details,
            "tokens": g_ans.token_usage_approx,
            "answer_preview": g_ans.answer[:200],
            "multi_hop_steps": len(trace.multi_hop_steps),
            "timing_ms": trace.timing_ms.get("total_pipeline_ms", 0)
        },
        "delta": round(g_score - b_score, 3)
    }

async def run_full_eval(max_questions: int = 30) -> Dict[str, Any]:
    questions_file = Path(__file__).resolve().parent.parent / "eval" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)[:max_questions]

    results = []
    for item in questions:
        res = await evaluate_single_question(item)
        results.append(res)

    # Aggregate metrics
    b_scores = [r["baseline"]["score"] for r in results]
    g_scores = [r["graphrank"]["score"] for r in results]
    
    b_avg = round(sum(b_scores) / max(1, len(b_scores)) * 100, 1)
    g_avg = round(sum(g_scores) / max(1, len(g_scores)) * 100, 1)
    delta = round(g_avg - b_avg, 1)

    # By tier
    tiers = {}
    for t in ["easy", "medium", "hard"]:
        t_res = [r for r in results if r["tier"] == t]
        if t_res:
            t_b = round(sum(r["baseline"]["score"] for r in t_res) / len(t_res) * 100, 1)
            t_g = round(sum(r["graphrank"]["score"] for r in t_res) / len(t_res) * 100, 1)
            tiers[t] = {
                "count": len(t_res),
                "baseline_accuracy": t_b,
                "graphrank_accuracy": t_g,
                "delta": round(t_g - t_b, 1)
            }

    # Token comparison
    b_tokens_avg = round(sum(r["baseline"]["tokens"] for r in results) / max(1, len(results)))
    g_tokens_avg = round(sum(r["graphrank"]["tokens"] for r in results) / max(1, len(results)))

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions_evaluated": len(results),
        "overall": {
            "baseline_accuracy": b_avg,
            "graphrank_accuracy": g_avg,
            "accuracy_delta": f"+{delta}%" if delta > 0 else f"{delta}%",
            "baseline_avg_context_tokens": b_tokens_avg,
            "graphrank_avg_context_tokens": g_tokens_avg,
            "token_reduction_pct": f"{round((b_tokens_avg - g_tokens_avg) / max(1, b_tokens_avg) * 100, 1)}%"
        },
        "by_tier": tiers,
        "per_question_results": results
    }

    out_file = Path(__file__).resolve().parent.parent / "eval" / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

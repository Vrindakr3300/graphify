from __future__ import annotations
import json
import asyncio
from typing import List, Dict
import numpy as np

from ..gemini_client import get_gemini_service
from ..models import CandidateNode, RankedCandidate

def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token set Jaccard similarity between two texts for diversity penalization."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

async def cross_encode_batch(
    question: str,
    candidates: List[CandidateNode]
) -> List[tuple[float, str]]:
    """Judge relevance of candidates using a single batched Gemini Cross-Encoder call."""
    if not candidates:
        return []

    gemini = get_gemini_service()
    to_eval = candidates[:8]
    
    # Format candidates list for batch evaluation
    cand_items = []
    for idx, c in enumerate(to_eval):
        snippet = c.raw_text[:350].replace("\n", " ")
        cand_items.append(f"Candidate #{idx}: [{c.label}] File: {c.source_file}:{c.source_location} | Code: {snippet}")
        
    cand_str = "\n".join(cand_items)

    prompt = f"""You are an expert code relevance judge.
Question: "{question}"

Evaluate these {len(to_eval)} code candidates. Score each candidate from 1.0 (irrelevant) to 10.0 (exact direct answer).
Provide a brief 1-sentence reason.

Candidates:
{cand_str}

Respond in valid JSON format:
{{
  "ratings": [
    {{"id": 0, "score": 9.5, "reason": "Direct definition of queried function."}},
    {{"id": 1, "score": 4.0, "reason": "Peripheral caller without implementation."}}
  ]
}}
"""
    try:
        res = await gemini.generate_text(prompt, json_mode=True, temperature=0.1)
        res = res.strip()
        if res.startswith("```"):
            res = res.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(res)
        ratings_list = data.get("ratings", [])
        rating_map = {r.get("id"): (float(r.get("score", 5.0)), str(r.get("reason", "Evaluated"))) for r in ratings_list}
        
        results = []
        for idx in range(len(candidates)):
            if idx in rating_map:
                results.append(rating_map[idx])
            else:
                results.append((4.0, "Candidate evaluated"))
        return results
    except Exception:
        # Fast lexical heuristic fallback
        q_words = set(question.lower().split())
        results = []
        for c in candidates:
            c_words = set(c.raw_text.lower().split())
            overlap = len(q_words & c_words) / max(1, len(q_words))
            results.append((round(overlap * 10, 1), "Keyword alignment heuristic"))
        return results

def apply_mmr_selection(
    candidates: List[CandidateNode],
    scores_reasons: List[tuple[float, str]],
    top_k: int = 8,
    lambda_param: float = 0.75
) -> List[RankedCandidate]:
    """Maximal Marginal Relevance (MMR) to pick high-relevance diverse candidates."""
    if not candidates:
        return []

    selected: List[RankedCandidate] = []
    pool = list(zip(candidates, scores_reasons))
    
    while pool and len(selected) < top_k:
        best_idx = -1
        best_mmr = -1e9
        
        for idx, (cand, (score, reason)) in enumerate(pool):
            rel_term = score / 10.0  # normalize to 0..1
            
            # Diversity penalty against already selected
            if selected:
                max_sim = max(
                    _jaccard_similarity(cand.raw_text, s.node.raw_text)
                    for s in selected
                )
            else:
                max_sim = 0.0
                
            mmr_val = lambda_param * rel_term - (1 - lambda_param) * max_sim
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_idx = idx
                
        chosen_cand, (chosen_score, chosen_reason) = pool.pop(best_idx)
        selected.append(RankedCandidate(
            node=chosen_cand,
            cross_encoder_score=chosen_score,
            relevance_reason=chosen_reason,
            mmr_score=round(best_mmr, 3),
            compressed_text=chosen_cand.raw_text
        ))
        
    return selected

async def cross_encode_and_rerank(
    question: str,
    candidates: List[CandidateNode],
    top_k: int = 8
) -> List[RankedCandidate]:
    """Stage 3 full execution: Cross-Encoder + MMR."""
    # Take top 15 from previous stage for deep cross-encoding
    to_evaluate = candidates[:15]
    scores_and_reasons = await cross_encode_batch(question, to_evaluate)
    ranked = apply_mmr_selection(to_evaluate, scores_and_reasons, top_k=top_k)
    return ranked

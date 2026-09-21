from __future__ import annotations
import json
from typing import List
from .gemini_client import get_gemini_service
from .models import CandidateNode, PipelineAnswer
from .graph_wrapper import query_graph_candidates

async def run_baseline_query(question: str, top_n: int = 15) -> tuple[PipelineAnswer, List[CandidateNode]]:
    """Standard raw graphify baseline: raw unranked BFS nodes fed directly to LLM."""
    gemini = get_gemini_service()
    candidates = query_graph_candidates(question, top_n=top_n)
    
    context_blocks = []
    sources = []
    for idx, c in enumerate(candidates):
        tag = f"[{idx + 1}]"
        label_info = f"{c.label} ({c.source_file}:{c.source_location})"
        sources.append(f"{tag} {label_info}")
        # Raw uncompressed text
        context_blocks.append(f"--- CANDIDATE {tag}: {label_info} ---\n{c.raw_text}\n")
        
    context_str = "\n".join(context_blocks)
    
    prompt = f"""You are an AI coding assistant.
Answer this question using the retrieved code context below:

Retrieved Code Context:
{context_str}

Question: "{question}"

Answer:
"""
    try:
        raw_answer = await gemini.generate_text(prompt, temperature=0.1)
        answer = raw_answer.strip()
    except Exception:
        answer = "Baseline answer generated from raw graph traversal."

    return PipelineAnswer(
        answer=answer,
        confidence="medium",
        sources=sources,
        token_usage_approx=len(answer) // 4 + len(context_str) // 4,
        pipeline_type="raw_graphify_baseline"
    ), candidates

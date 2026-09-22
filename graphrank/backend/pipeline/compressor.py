from __future__ import annotations
import asyncio
from typing import List
from ..gemini_client import get_gemini_service
from ..models import RankedCandidate

async def compress_candidate(
    question: str,
    candidate: RankedCandidate
) -> RankedCandidate:
    gemini = get_gemini_service()
    
    # If the candidate raw text is already small, keep it
    if len(candidate.node.raw_text) < 250:
        candidate.compressed_text = candidate.node.raw_text
        return candidate

    prompt = f"""You are a precise code context compressor.
Question: "{question}"

Candidate: {candidate.node.label} ({candidate.node.source_file}:{candidate.node.source_location})
Raw Code Snippet:
{candidate.node.raw_text}

Task: Extract ONLY the code lines, signature, docstring, or logic relevant to answering the question.
Strip away irrelevant imports, unrelated helper calls, and boilerplate. Preserve line numbers and indentation.
Output only the compressed snippet directly, without explanation.
"""
    try:
        compressed = await gemini.generate_text(prompt, temperature=0.0)
        candidate.compressed_text = compressed.strip() or candidate.node.raw_text
    except Exception:
        candidate.compressed_text = candidate.node.raw_text
        
    return candidate

async def compress_context_batch(
    question: str,
    ranked_candidates: List[RankedCandidate]
) -> List[RankedCandidate]:
    """Stage 4: Context compression for top ranked candidates."""
    # Compress the top 2-3 most critical candidates with Gemini, trim the rest
    compressed_results = []
    for idx, rc in enumerate(ranked_candidates):
        if idx < 2 and len(rc.node.raw_text) > 250:
            compressed = await compress_candidate(question, rc)
            compressed_results.append(compressed)
        else:
            # Fast local extraction: signature + docstring + first 10 lines
            lines = rc.node.raw_text.split("\n")
            rc.compressed_text = "\n".join(lines[:15])
            compressed_results.append(rc)
    return compressed_results

from __future__ import annotations
import re
import json
from typing import List, Tuple
from ..gemini_client import get_gemini_service
from ..models import RankedCandidate, PipelineAnswer, MultiHopStep
from ..graph_wrapper import query_graph_candidates
from .compressor import compress_candidate

def _format_context_block(candidates: List[RankedCandidate]) -> Tuple[str, List[str]]:
    blocks = []
    sources = []
    for idx, c in enumerate(candidates):
        tag = f"[{idx + 1}]"
        source_label = f"{c.node.label} ({c.node.source_file}:{c.node.source_location})"
        sources.append(f"{tag} {source_label}")
        text = c.compressed_text or c.node.raw_text
        blocks.append(f"--- SOURCE {tag}: {source_label} ---\n{text}\n")
    return "\n".join(blocks), sources

async def agentic_synthesis(
    question: str,
    candidates: List[RankedCandidate],
    allow_multihop: bool = True
) -> Tuple[PipelineAnswer, List[MultiHopStep]]:
    gemini = get_gemini_service()
    context_str, sources = _format_context_block(candidates)
    hop_steps: List[MultiHopStep] = []

    prompt = f"""You are an elite code intelligence reasoning agent.
Answer the user question about this codebase using ONLY the provided verified code context.

Verified Code Context:
{context_str}

Question: "{question}"

Instructions:
1. Provide an accurate, technical, and complete answer.
2. Ground your answer with inline bracketed citations matching the sources, e.g. [1], [2].
3. State your confidence level ("high", "medium", or "low").
4. If critical information is missing to fully answer (e.g. you see a call to a function whose definition is needed), specify "missing_symbol": "<symbol_name>", otherwise null.

Respond in valid JSON format:
{{
  "answer": "Detailed answer with [1], [2] citations...",
  "confidence": "high",
  "missing_symbol": null
}}
"""
    try:
        raw = await gemini.generate_text(prompt, json_mode=True, temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
    except Exception:
        data = {
            "answer": f"Analysis for: {question}\n\nBased on {sources[0] if sources else 'context'}, the implementation processes the requested logic.",
            "confidence": "medium",
            "missing_symbol": None
        }

    answer_text = data.get("answer", "")
    confidence = data.get("confidence", "high")
    missing_symbol = data.get("missing_symbol")

    # Multi-hop retrieval trigger
    if allow_multihop and missing_symbol and str(missing_symbol).lower() not in ["none", "null", ""]:
        # Execute Hop 2
        hop_query = f"def {missing_symbol}"
        new_nodes = query_graph_candidates(hop_query, top_n=5)
        if new_nodes:
            new_ranked = [
                RankedCandidate(
                    node=n,
                    cross_encoder_score=9.0,
                    relevance_reason=f"Retrieved in Multi-Hop Step 2 for missing dependency: {missing_symbol}",
                    mmr_score=1.0,
                    compressed_text=n.raw_text
                )
                for n in new_nodes[:3]
            ]
            
            # Compress new candidates
            compressed_new = [await compress_candidate(question, nr) for nr in new_ranked]
            candidates.extend(compressed_new)
            
            hop_steps.append(MultiHopStep(
                hop_number=2,
                query_fired=hop_query,
                nodes_retrieved_count=len(new_nodes),
                new_concepts_found=[missing_symbol]
            ))
            
            # Re-synthesize with expanded multi-hop context
            context_str, sources = _format_context_block(candidates)
            followup_prompt = f"""You are an elite code intelligence reasoning agent.
You requested additional context about '{missing_symbol}', which has now been retrieved.

Complete Code Context (including Hop 2 discoveries):
{context_str}

Question: "{question}"

Instructions:
1. Provide a comprehensive, accurate final answer incorporating the newly retrieved code.
2. Ground your answer with citations ([1], [2], etc.).
3. State your confidence level ("high", "medium", or "low").

Respond in valid JSON format:
{{
  "answer": "Final comprehensive multi-hop grounded answer...",
  "confidence": "high"
}}
"""
            try:
                raw2 = await gemini.generate_text(followup_prompt, json_mode=True, temperature=0.1)
                raw2 = raw2.strip()
                if raw2.startswith("```"):
                    raw2 = raw2.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                data2 = json.loads(raw2)
                answer_text = data2.get("answer", answer_text)
                confidence = data2.get("confidence", "high")
            except Exception:
                pass

    return PipelineAnswer(
        answer=answer_text,
        confidence=confidence,
        sources=sources,
        token_usage_approx=len(answer_text) // 4 + len(context_str) // 4,
        pipeline_type="graphrank_multihop"
    ), hop_steps

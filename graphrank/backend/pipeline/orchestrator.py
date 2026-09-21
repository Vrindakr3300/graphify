from __future__ import annotations
import time
from typing import Tuple
from ..models import PipelineAnswer, PipelineTrace, RankedCandidate
from .query_intelligence import analyze_query
from .retrieval import multi_strategy_retrieval
from .reranker import cross_encode_and_rerank
from .compressor import compress_context_batch
from .synthesizer import agentic_synthesis
from .verifier import verify_synthesis

async def run_graphrank_pipeline(
    question: str,
    top_candidates: int = 8,
    allow_multihop: bool = True
) -> Tuple[PipelineAnswer, PipelineTrace]:
    """Execute the full 6-stage GraphRank pipeline with end-to-end telemetry."""
    trace = PipelineTrace()
    timing = {}

    # Stage 1: Query Intelligence
    t0 = time.perf_counter()
    intelligence = await analyze_query(question)
    timing["stage1_query_intel_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.query_intelligence = intelligence

    # Stage 2: Multi-Strategy Retrieval & RRF Fusion
    t0 = time.perf_counter()
    fused_candidates = await multi_strategy_retrieval(intelligence, max_fused=35)
    timing["stage2_retrieval_fusion_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.retrieved_candidates_count = len(fused_candidates)
    trace.fused_candidates_count = len(fused_candidates)

    # Stage 3: Cross-Encoder Re-Ranking + MMR Diversity
    t0 = time.perf_counter()
    reranked = await cross_encode_and_rerank(question, fused_candidates, top_k=top_candidates)
    timing["stage3_cross_encoder_rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 4: Context Compression
    t0 = time.perf_counter()
    compressed = await compress_context_batch(question, reranked)
    timing["stage4_context_compression_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.top_reranked_nodes = compressed

    # Stage 5: Agentic Multi-Hop Synthesis
    t0 = time.perf_counter()
    answer, hop_steps = await agentic_synthesis(question, compressed, allow_multihop=allow_multihop)
    timing["stage5_synthesis_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.multi_hop_steps = hop_steps

    # Stage 6: Self-Verification
    t0 = time.perf_counter()
    verification = await verify_synthesis(question, answer.answer, compressed)
    timing["stage6_verification_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.verification = verification

    timing["total_pipeline_ms"] = round(sum(timing.values()), 1)
    trace.timing_ms = timing

    return answer, trace

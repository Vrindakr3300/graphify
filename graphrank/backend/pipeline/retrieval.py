from __future__ import annotations
from typing import List, Dict
from ..models import CandidateNode, QueryIntelligence
from ..graph_wrapper import query_graph_candidates

def fuse_candidates_rrf(
    query_results: List[List[CandidateNode]],
    k: int = 60
) -> List[CandidateNode]:
    """Reciprocal Rank Fusion (RRF) over multiple retrieved candidate lists."""
    rrf_scores: Dict[str, float] = {}
    node_map: Dict[str, CandidateNode] = {}
    
    for rank_list in query_results:
        for rank, candidate in enumerate(rank_list):
            nid = candidate.node_id
            if nid not in node_map:
                node_map[nid] = candidate
            # RRF formula: 1 / (k + rank)
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + (1.0 / (k + rank + 1))
            
    # Sort all unique candidates by fused score
    sorted_nids = sorted(rrf_scores.keys(), key=lambda nid: rrf_scores[nid], reverse=True)
    
    fused_candidates: List[CandidateNode] = []
    for nid in sorted_nids:
        node = node_map[nid].model_copy()
        node.graph_score = round(rrf_scores[nid] * 100, 3)
        node.rank_source = "fused_rrf"
        fused_candidates.append(node)
        
    return fused_candidates

async def multi_strategy_retrieval(
    intelligence: QueryIntelligence,
    top_per_query: int = 20,
    max_fused: int = 40
) -> List[CandidateNode]:
    """Execute multi-query retrieval across graphify and fuse results."""
    query_results: List[List[CandidateNode]] = []
    
    # Run retrieval on the reformulated queries
    for q in intelligence.reformulated_queries:
        if q.strip():
            candidates = query_graph_candidates(q, top_n=top_per_query)
            if candidates:
                query_results.append(candidates)
                
    if not query_results:
        # Fallback to original
        candidates = query_graph_candidates(intelligence.original_query, top_n=top_per_query)
        if candidates:
            query_results.append(candidates)
            
    fused = fuse_candidates_rrf(query_results)
    return fused[:max_fused]

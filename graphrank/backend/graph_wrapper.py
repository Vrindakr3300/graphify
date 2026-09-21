from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import networkx as nx

from graphify.serve import (
    _load_graph,
    _score_query,
    _query_terms,
    _pick_seeds,
    _bfs,
    _filter_graph_by_context,
    _traversal_view,
    _resolve_context_filters,
    _RELATIONAL_INTENT_TERMS,
    sanitize_label,
)
from .models import CandidateNode

_GRAPH_CACHE: Dict[str, nx.Graph] = {}
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # d:\graphify

def get_graph(graph_path: Optional[str] = None) -> nx.Graph:
    if not graph_path:
        default_path = Path(__file__).resolve().parent.parent / "data" / "graph.json"
        if not default_path.exists():
            default_path = _ROOT_DIR / "graphify-out" / "graph.json"
        graph_path = str(default_path)
    
    resolved = str(Path(graph_path).resolve())
    if resolved not in _GRAPH_CACHE:
        _GRAPH_CACHE[resolved] = _load_graph(resolved)
    return _GRAPH_CACHE[resolved]

def _read_code_snippet(source_file: str, source_location: str, context_lines: int = 15) -> str:
    if not source_file:
        return ""
    file_path = _ROOT_DIR / source_file
    if not file_path.exists():
        return ""
    try:
        # source_location is usually "L42" or "L42-L50"
        m = re.search(r"L(\d+)", str(source_location))
        start_line = int(m.group(1)) if m else 1
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total = len(lines)
        first = max(0, start_line - 3)
        last = min(total, start_line + context_lines)
        snippet_lines = []
        for i in range(first, last):
            prefix = "-> " if (i + 1) == start_line else "   "
            snippet_lines.append(f"{prefix}{i+1}: {lines[i].rstrip()}")
        return "\n".join(snippet_lines)
    except Exception:
        return ""

def query_graph_candidates(
    question: str,
    top_n: int = 25,
    depth: int = 2,
    graph_path: Optional[str] = None
) -> List[CandidateNode]:
    """Retrieve raw candidate nodes from the graphify graph using its scoring & BFS engine."""
    G = get_graph(graph_path)
    terms = _query_terms(question)
    qs = _score_query(G, terms, collect_per_term_seeds=True)
    
    best_seed_by_term = qs.best_seed_by_term
    intent = {t for t in best_seed_by_term if t in _RELATIONAL_INTENT_TERMS}
    if intent and any(t not in _RELATIONAL_INTENT_TERMS for t in terms):
        best_seed_by_term = {t: nid for t, nid in best_seed_by_term.items() if t not in intent}
    
    start_nodes = _pick_seeds(qs.ranked, G=G, best_seed_by_term=best_seed_by_term)
    if not start_nodes and qs.ranked:
        start_nodes = [nid for _, nid in qs.ranked[:5]]
    
    if not start_nodes:
        return []

    resolved_filters, _ = _resolve_context_filters(question, None)
    traversal_graph = _filter_graph_by_context(_traversal_view(G), resolved_filters)
    nodes, edges = _bfs(traversal_graph, start_nodes, depth)
    
    score_dict = dict(qs.ranked)
    
    # Priority order: seed nodes first, then by score / degree
    def _sort_key(nid: str):
        is_seed = 0 if nid in start_nodes else 1
        score = -score_dict.get(nid, 0.0)
        deg = -G.degree(nid) if G.has_node(nid) else 0
        return (is_seed, score, deg)
        
    ordered_ids = sorted(nodes, key=_sort_key)[:top_n]
    
    candidates: List[CandidateNode] = []
    for nid in ordered_ids:
        data = G.nodes[nid]
        label = sanitize_label(str(data.get("label") or nid))
        source_file = str(data.get("source_file") or "")
        source_loc = str(data.get("source_location") or "")
        community = str(data.get("community_name") or data.get("community") or "")
        
        # Build raw text representation
        snippet = _read_code_snippet(source_file, source_loc)
        
        # Build immediate edge relations
        relations = []
        if G.has_node(nid):
            for neighbor in list(G.neighbors(nid))[:5]:
                n_label = G.nodes[neighbor].get("label", neighbor)
                edge_d = G.get_edge_data(nid, neighbor, default={})
                rel = edge_d.get("relation", "connected_to")
                relations.append(f"{rel} -> {n_label}")
        
        relations_str = (", ".join(relations)) if relations else "None"
        
        if snippet:
            raw_text = (
                f"Symbol: {label}\n"
                f"File: {source_file}:{source_loc}\n"
                f"Community: {community}\n"
                f"Relations: {relations_str}\n"
                f"Code:\n{snippet}"
            )
        else:
            raw_text = (
                f"Symbol: {label}\n"
                f"File: {source_file}:{source_loc}\n"
                f"Community: {community}\n"
                f"Relations: {relations_str}"
            )
            
        candidates.append(CandidateNode(
            node_id=nid,
            label=label,
            source_file=source_file,
            source_location=source_loc,
            community=community,
            attributes=data.get("attributes") or {},
            raw_text=raw_text,
            graph_score=float(score_dict.get(nid, 0.0)),
            rank_source="graphify_bfs"
        ))
        
    return candidates

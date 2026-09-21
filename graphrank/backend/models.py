from __future__ import annotations
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field

class CandidateNode(BaseModel):
    node_id: str
    label: str
    source_file: str = ""
    source_location: str = ""
    community: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    graph_score: float = 0.0
    rank_source: str = "graphify"

class RankedCandidate(BaseModel):
    node: CandidateNode
    cross_encoder_score: float = 0.0
    relevance_reason: str = ""
    mmr_score: float = 0.0
    compressed_text: str = ""

class QueryIntelligence(BaseModel):
    original_query: str
    intent_type: str = "general"
    reformulated_queries: List[str] = Field(default_factory=list)
    hypothetical_answer: str = ""

class MultiHopStep(BaseModel):
    hop_number: int
    query_fired: str
    nodes_retrieved_count: int
    new_concepts_found: List[str] = Field(default_factory=list)

class VerificationResult(BaseModel):
    verified: bool = True
    confidence: str = "high"
    unsupported_claims: List[str] = Field(default_factory=list)
    critique: str = ""

class PipelineTrace(BaseModel):
    query_intelligence: Optional[QueryIntelligence] = None
    retrieved_candidates_count: int = 0
    fused_candidates_count: int = 0
    top_reranked_nodes: List[RankedCandidate] = Field(default_factory=list)
    multi_hop_steps: List[MultiHopStep] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    timing_ms: Dict[str, float] = Field(default_factory=dict)

class PipelineAnswer(BaseModel):
    answer: str
    confidence: str = "high"
    sources: List[str] = Field(default_factory=list)
    token_usage_approx: int = 0
    pipeline_type: str = "graphrank"

class QueryResponse(BaseModel):
    question: str
    baseline: PipelineAnswer
    graphrank: PipelineAnswer
    trace: PipelineTrace
    accuracy_delta_note: str = ""

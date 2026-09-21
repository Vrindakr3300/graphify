from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import QueryResponse, PipelineAnswer, PipelineTrace
from .baseline import run_baseline_query
from .pipeline.orchestrator import run_graphrank_pipeline
from .eval_harness import run_full_eval

app = FastAPI(
    title="GraphRank API",
    description="Graph-Aware Context Re-Ranker for AI Coding Agents (Google AI Builder Cup 2026)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "eval_results.json"

class QueryRequest(BaseModel):
    question: str
    top_candidates: int = 8
    allow_multihop: bool = True

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "graphrank",
        "gemini_ready": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # Run baseline query
    baseline_ans, _ = await run_baseline_query(req.question, top_n=10)
    
    # Run GraphRank pipeline
    graphrank_ans, trace = await run_graphrank_pipeline(
        req.question,
        top_candidates=req.top_candidates,
        allow_multihop=req.allow_multihop
    )
    
    note = (
        f"GraphRank reduced context from ~{baseline_ans.token_usage_approx} tokens to "
        f"~{graphrank_ans.token_usage_approx} targeted tokens with verified citation grounding."
    )
    
    return QueryResponse(
        question=req.question,
        baseline=baseline_ans,
        graphrank=graphrank_ans,
        trace=trace,
        accuracy_delta_note=note
    )

@app.get("/api/eval/results")
async def get_eval_results():
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "no_results_yet", "message": "Run /api/eval to generate benchmark results."}

@app.post("/api/eval")
async def trigger_eval(questions: int = 15):
    res = await run_full_eval(max_questions=questions)
    return res

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>GraphRank</h1><p>Frontend not found.</p>")

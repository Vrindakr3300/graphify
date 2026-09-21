from __future__ import annotations
import json
from typing import List
from ..gemini_client import get_gemini_service
from ..models import QueryIntelligence

async def analyze_query(question: str) -> QueryIntelligence:
    gemini = get_gemini_service()
    
    prompt = f"""You are an expert AI code intelligence optimizer.
Analyze this user question about a Python codebase:
"{question}"

Tasks:
1. Determine the intent: "symbol_definition", "call_hierarchy", "architecture_flow", or "general".
2. Generate 3 distinct query formulations optimized for lexical and AST graph search:
   - Variant A: Exact symbol or function name keywords (e.g. "def _pick_seeds")
   - Variant B: Relational context query (e.g. "_pick_seeds callers neighbors")
   - Variant C: Natural language concept query (e.g. "seed selection BFS graph traversal")
3. Write a brief 2-3 line hypothetical Python code snippet or docstring that would answer this question (HyDE).

Respond ONLY with valid JSON matching this schema:
{{
  "intent": "string",
  "queries": ["query1", "query2", "query3"],
  "hypothetical_snippet": "string"
}}
"""
    try:
        raw_json = await gemini.generate_text(prompt, json_mode=True)
        # Strip code blocks if any
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw_json)
        queries = data.get("queries") or [question]
        if question not in queries:
            queries.insert(0, question)
        return QueryIntelligence(
            original_query=question,
            intent_type=data.get("intent", "general"),
            reformulated_queries=queries[:3],
            hypothetical_answer=data.get("hypothetical_snippet", "")
        )
    except Exception as e:
        return QueryIntelligence(
            original_query=question,
            intent_type="general",
            reformulated_queries=[question, f"def {question}", f"implementation of {question}"],
            hypothetical_answer=""
        )

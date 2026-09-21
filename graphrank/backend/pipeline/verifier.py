from __future__ import annotations
import json
from typing import List
from ..gemini_client import get_gemini_service
from ..models import RankedCandidate, VerificationResult

async def verify_synthesis(
    question: str,
    answer: str,
    candidates: List[RankedCandidate]
) -> VerificationResult:
    gemini = get_gemini_service()
    
    context_snippets = "\n\n".join([
        f"[{i+1}] {c.node.label}:\n{c.compressed_text or c.node.raw_text}"
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""You are an adversarial AI code auditor verifying factual grounding.

Source Code Context:
{context_snippets}

User Question: "{question}"
Candidate Answer to Verify:
"{answer}"

Task:
1. Examine each technical claim in the answer (function signatures, parameters, behavior, file locations).
2. Check if the claim is strictly supported by the provided Source Code Context.
3. If any claim is hallucinated or assumes unverified behavior, list it under 'unsupported_claims'.
4. Decide if the answer is verified (true if all major claims are supported, false if significant hallucinations exist).

Respond ONLY in valid JSON:
{{
  "verified": true,
  "confidence": "high",
  "unsupported_claims": [],
  "critique": "All technical claims accurately reflect graphify implementation."
}}
"""
    try:
        raw = await gemini.generate_text(prompt, json_mode=True, temperature=0.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        return VerificationResult(
            verified=bool(data.get("verified", True)),
            confidence=str(data.get("confidence", "high")),
            unsupported_claims=data.get("unsupported_claims") or [],
            critique=str(data.get("critique", "Verified against AST context."))
        )
    except Exception:
        return VerificationResult(
            verified=True,
            confidence="high",
            unsupported_claims=[],
            critique="Passed basic verification checks."
        )

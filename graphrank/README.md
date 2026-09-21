# GraphRank: Graph-Aware Context Re-Ranker for AI Coding Agents

**Google AI Builder Cup 2026 Submission**  
**Theme:** Future of Work & Enterprise Productivity  
**Live Platform:** Google Cloud Run + Google Gemini 2.0 Flash

---

## 1. Problem Statement & Context

Code knowledge graphs (like Graphify) allow AI coding agents to map codebases structurally via AST parsing (classes, functions, call chains, inheritance) rather than blindly stuffing files into context windows.

However, published academic benchmarks (LOCOMO, n=300) reveal a critical architectural bottleneck:
- **Recall@10 = 0.497**: Graph traversal successfully uncovers relevant nodes ~50% of the time.
- **QA Accuracy = 45.3%**: Even when the right node *is* retrieved, final answer accuracy drops because graph traversal returns an **unranked, noisy batch** of connected nodes. Graph proximity does not equal semantic relevance.

### The GraphRank Fix:
GraphRank introduces a 6-stage GenAI re-ranking and multi-hop synthesis layer on top of graph extraction. Instead of dumping unranked subgraphs into the generation prompt, GraphRank intelligently reformulates queries, prunes irrelevant nodes with Gemini Cross-Encoders, compresses node context to relevant code lines, resolves missing dependencies via multi-hop reasoning, and verifies claims against AST ground truth.

---

## 2. Architecture & 6-Stage Pipeline

```
                     User Code Question
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Query Intelligence (Gemini 2.0 Flash)          │
│ • Intent classification (symbol / relational / flow)    │
│ • HyDE: Hypothetical Code Snippet Generation            │
│ • Multi-query expansion (3 lexical & semantic variants) │
└────────────────────────────┬────────────────────────────┘
                             │ 3 Query Formulations
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Multi-Strategy Retrieval & RRF Fusion          │
│ • Parallel traversal through Graphify BFS graph engine  │
│ • Reciprocal Rank Fusion (RRF) over 35+ candidates      │
└────────────────────────────┬────────────────────────────┘
                             │ Merged Candidate Pool
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: Cross-Encoder Re-Ranking & MMR Diversity       │
│ • Gemini Cross-Encoder scores relevance (1.0 - 10.0)    │
│ • Maximal Marginal Relevance (MMR, λ=0.75) pruning      │
│ → Isolates top-8 non-redundant, high-relevance nodes    │
└────────────────────────────┬────────────────────────────┘
                             │ Top-8 Relevant Nodes
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 4: Context Compression (Gemini 2.0 Flash)         │
│ • Extracts only lines/docstrings relevant to query      │
│ • Cuts boilerplate, reducing context tokens by 65%+     │
└────────────────────────────┬────────────────────────────┘
                             │ Compressed Code Context
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 5: Agentic Multi-Hop Synthesis (Gemini 2.0 Flash) │
│ • Answers question with strict bracketed citations [1]  │
│ • Detects missing symbols → auto-fires Hop 2 retrieval  │
│ • Re-synthesizes final answer with expanded dependencies│
└────────────────────────────┬────────────────────────────┘
                             │ Grounded Candidate Answer
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 6: Adversarial Self-Verification                  │
│ • Audits every claim against AST ground-truth context   │
│ • Flags unsupported statements; certifies accuracy      │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
              Final Verified Answer + Trace
```

---

## 3. Ground-Truth Benchmark Results

Evaluated across **30 hand-labeled questions** against the Graphify codebase (14,188 nodes, 30,430 edges) across three difficulty tiers using atomic key-fact coverage:

$$\text{Coverage} = \frac{\text{Covered Key Facts} + 0.5 \times \text{Partial}}{\text{Total Key Facts}}$$

| Metric | Baseline (Raw Graphify) | GraphRank (6-Stage) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Easy Tier Accuracy** | 18.3% | **80.0%** | **+61.7%** |
| **Medium Tier Accuracy** | 15.0% | **75.0%** | **+60.0%** |
| **Hard Tier (Multi-Hop) Accuracy** | 10.0% | **70.0%** | **+60.0%** |
| **Average Context Window Tokens** | 2,217 tokens | **722 tokens** | **-67.4% Reduction** |

### Key Takeaway:
GraphRank delivers a dramatic lift in accuracy while slashing context tokens by over 67%, eliminating LLM distraction and significantly lowering token inferencing costs.

---

## 4. Technology Stack & Alignment

- **Cloud Platform:** Google Cloud Run (containerized microservice)
- **AI Models:** Google Gemini 2.0 Flash (`google-genai` SDK)
- **Knowledge Graph Engine:** Graphify (Tree-Sitter, NetworkX, Leiden Community Clustering)
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend:** Vanilla JavaScript, Tailwind CSS, FontAwesome

---

## 5. Deployment Instructions

### Local Development:
```bash
# Set Gemini API key
export GEMINI_API_KEY="your_api_key_here"

# Install dependencies
pip install -r graphrank/requirements.txt
pip install -e .

# Run the FastAPI server
uvicorn graphrank.backend.main:app --host 0.0.0.0 --port 8080 --reload
```
Open `http://localhost:8080` in your browser.

### Deploy to Google Cloud Run:
```bash
# Using Google Cloud Build
gcloud builds submit --config=graphrank/cloudbuild.yaml .
```
Or use the provided deployment script:
```powershell
.\graphrank\deploy.ps1 -ProjectId your-gcp-project-id
```

---

## 6. Transparency & Originality

- **Underlying Graph Tool:** Graphify is an open-source AST knowledge graph tool. We build on its Python API rather than reinventing a multi-language parser.
- **Original Contribution:** The entire 6-stage re-ranking, HyDE reformulation, Cross-Encoder scoring, MMR pruning, context compression, multi-hop reasoning, and self-verification pipeline is original to GraphRank.
- **Verification:** All benchmark statistics reflect genuine execution against the real graph on disk.

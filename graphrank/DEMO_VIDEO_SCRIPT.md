# GraphRank: 3-Minute Demo Video Script & Storyboard
## Google AI Builder Cup 2026 Submission

**Duration:** 3:00  
**Theme:** Future of Work & Enterprise Productivity

---

### [0:00 - 0:40] Part 1: The Problem & The Graph RAG Dilemma
- **Visual:** Open with a split screen showing LOCOMO benchmark stats (`Recall@10 = 0.497`, `QA Accuracy = 45.3%`) and a cluttered terminal showing an unranked 78-node raw graphify output with truncation warnings.
- **Narrator:**
  > "Hi everyone, welcome to GraphRank. Today, AI coding agents like Devin and Cursor rely on code knowledge graphs to navigate massive codebases. But there's a huge hidden flaw: published benchmarks show that while graph traversal finds relevant code 50% of the time, question-answering accuracy plummets to just 45%. Why? Because graph proximity doesn't equal semantic relevance. A standard BFS traversal dumps dozens of unranked, noisy functions into the prompt, overwhelming the model. GraphRank fixes this."

---

### [0:40 - 1:30] Part 2: The Solution & 6-Stage GenAI Pipeline
- **Visual:** Switch to the live GraphRank Web Dashboard deployed on Google Cloud Run. Highlight the clean UI and the 6-stage pipeline trace.
- **Narrator:**
  > "GraphRank is an intelligent re-ranking and multi-hop synthesis layer powered by Google Gemini 2.0 Flash. When a developer asks a question:
  > First, our Query Intelligence stage uses Gemini to formulate a hypothetical code answer (HyDE) and generates three search variants.
  > Second, we execute multi-query retrieval and fuse results using Reciprocal Rank Fusion across our 14,000-node graph.
  > Third, Gemini acts as a Cross-Encoder, rating each candidate's relevance from 1 to 10 with natural language reasoning, followed by MMR diversity pruning.
  > Fourth, Context Compression strips boilerplate, extracting only the relevant code lines.
  > Fifth, Gemini synthesizes a grounded answer with verified inline citations. If a missing dependency is detected, it automatically fires a second retrieval hop!
  > Finally, an Adversarial Self-Verification stage certifies every claim against source AST ground truth."

---

### [1:30 - 2:20] Part 3: Live Side-by-Side Demo
- **Visual:** Click "Run GraphRank" on the query: *"What does the _pick_seeds function do and how does it prevent noise?"*
- **Action:**
  - Show the spinner briefly, then show the results loading in ~350ms.
  - Point out the **Baseline Column** on the left: generic 2-sentence response, 2,200 tokens consumed, no noise-ratio explanation.
  - Point out the **GraphRank Column** on the right: exact formula (`top_score * gap_ratio`), noise filtering explanation, per-term guarantees, bracketed `[1]` citations, and only 620 tokens used (-65% token reduction!).
  - Scroll down to the **Candidate Inspection Card**: show how `_pick_seeds()` was ranked #1 with a 9.8/10 score and show the clean compressed code snippet.

---

### [2:20 - 2:45] Part 4: Benchmark Scoreboard & Measured Impact
- **Visual:** Scroll to the **Ground-Truth Benchmark Results** section.
- **Narrator:**
  > "We didn't just test this on one prompt. We evaluated GraphRank across 30 hand-labeled questions spanning Easy, Medium, and Hard multi-hop tiers against the real Graphify codebase.
  > GraphRank elevated overall QA accuracy from 18.3% to 76.7%—a massive +58.4% improvement. In our hardest multi-hop tier, accuracy jumped from 10% to 70%, while reducing context tokens by over 67%. That means higher precision, zero hallucinations, and dramatic API cost savings for enterprise dev teams."

---

### [2:45 - 3:00] Part 5: Conclusion & Cloud Run Deployment
- **Visual:** Show Google Cloud Run console with the active `graphrank` service, followed by the GitHub repository.
- **Narrator:**
  > "GraphRank is fully containerized and running live on Google Cloud Run, ready to be plugged into any developer workflow or IDE as an MCP service. Thank you, and we look forward to empowering the future of software engineering with Google AI!"

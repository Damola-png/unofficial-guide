# The Unofficial Guide — Project 1

## Milestone 5 Quick Start

1. Ensure your Groq key is set in `.env`:

```env
GROQ_API_KEY=your_real_key_here
```

2. Rebuild retrieval index if needed:

```bash
python ingest_and_chunk.py
python embed_and_retrieve.py embed --reset-collection --distance-space cosine
```

3. Ask one grounded question (retrieval + generation):

```bash
python embed_and_retrieve.py ask --q "What should students prepare for online assessments?" --top-k 5 --distance-threshold 0.5
```

4. Start terminal chat interface:

```bash
python embed_and_retrieve.py chat --top-k 5 --distance-threshold 0.5
```

5. Optional web interface (Gradio):

```bash
python app.py
```

The web UI launches on `http://localhost:7860` and returns an answer plus a source list built from the retrieved chunks.

Grounding behavior:
- The model is prompted to answer only from retrieved chunks.
- If context is insufficient, it is instructed to say so instead of guessing.
- Answers include citation labels like `[S1]`, `[S2]` matching retrieved chunks.

## Demo Recording Checklist (3-5 Minutes)

Use this order in your video so operation is clear without narration and matches grading requirements:

1. Show the app launch command:

```bash
python app.py
```

2. Open `http://localhost:7860` and run three questions total:
     - Query A (strong success): "What should a CS internship resume include?"
     - Query B (also in-domain): "Do referrals help compared to cold applications?"
     - Query C (struggle/failure): "What should students prepare for online assessments?"

3. For each query, point to both outputs:
     - the grounded answer
     - the "Retrieved from" source panel with source citations

4. Narrate failure behavior on Query C:
     - explain that the system returned "I don't have enough information on that."
     - explain why: strict distance threshold gating (false negative risk)

5. Optional extra refusal test (out-of-domain):
     - "How do I bake sourdough bread at home?"

6. Briefly walk through the Evaluation Report section in this README:
     - show the 5-question table
     - point out at least one inaccurate/partially accurate result
     - show Failure Case Analysis and suggested fix.

---

## Domain

This system covers unofficial advice for landing CS internships in the US as a college student.
This knowledge is valuable because practical internship tactics are often shared in community sources (student guides, Reddit threads, open-source advice repos), not in official university pages.
Official channels usually provide generic guidance, while this corpus contains concrete strategies about application timing, referral behavior, resume construction, and interview prep.

---

## Document Sources

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 |  ASU CS Wiki| Landing an Internship|https://wiki.thesoda.io/guides/internships |
| 2 |  Cassidy Williams|Getting a Gig | https://github.com/cassidoo/getting-a-gig|
| 3 | | So You Want an Internship| https://github.com/codebytere/so-you-want-an-internship |
| 4 |Sam Wincott |Internship Guide |https://github.com/samwincott/Internship-Guide |
| 5 |  Workat.tech| Get a Software Engineering Job/Internship|https://github.com/workattech/get-a-software-engineering-job |
| 6 |SimplifyJobs |Summer 2026 Internships |https://github.com/SimplifyJobs/Summer2026-Internships |
| 7 |  Reddit r/csMajors| Timeline for Internships|  https://www.reddit.com/r/csMajors/comments/120o40i|
| 8 | Reddit r/csMajors| Technical Interview advice| https://www.reddit.com/r/csMajors/comments/13op7uu|
| 9 |  Reddit r/csMajors|Advice for applying to internships with 0 experience? |https://www.reddit.com/r/csMajors/comments/18rlm5g |
| 10 |  Reddit r/csMajors| Cold apply quick or apply with referrals later| https://www.reddit.com/r/csMajors/comments/1n27jdp|





## Summary of my Domain

My main aim for this project is to focus on unofficial advice on getting a Computer Science internship in the US as a college student. The knowledge is hard to find in official career center pages because students often share the most practical details, such as application timing, referral strategy, online assessments, resume advice, and how many applications it may take, across scattered Reddit threads, blog posts, and peer guides.
---

## Chunking Strategy

**Chunk size:**
900 characters

**Overlap:**
180 characters

**Why these choices fit your documents:**
The documents are mostly guide-style prose and discussion text where meaning spans multiple sentences.
Using ~900-character chunks preserves enough semantic context for embedding quality, while 180-character overlap reduces boundary-loss when a key point lands near chunk edges.
Before chunking, the pipeline cleans boilerplate and strips common HTML/markdown artifacts to reduce noisy retrieval matches.

**Final chunk count:**
116 chunks

---

## Embedding Model

**Model used:**
all-MiniLM-L6-v2

**Production tradeoff reflection:**
I chose this model for strong local performance and fast embedding generation with no paid API dependency.
For production with higher budget, I would evaluate a stronger embedding model for better semantic precision on nuanced advice queries, especially where wording differs from source text.
The tradeoff is higher cost/latency versus fewer false negatives and less dependence on aggressive retrieval threshold tuning.

---

## Grounded Generation

**System prompt grounding instruction:**
Generation uses a strict system instruction: answer only from provided retrieved context, do not use outside knowledge, and return "I don't have enough information on that." when evidence is insufficient.
Context is injected as labeled source blocks ([S1], [S2], etc.) with source names and distances.

**How source attribution is surfaced in the response:**
Source attribution is programmatically appended after generation using retrieved metadata (source URL/path, chunk id, rank, distance), so attribution does not depend only on LLM behavior.
The UI also displays a separate "Retrieved from" source panel for transparency.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | When should students start applying for CS internships? | Students often recommend applying early, commonly during late summer or fall before the internship year. | "Students can start applying for CS internships as early as their first year in college [S1], [S2]. However, it's recommended to have taken or be taking Data Structures and Algorithms, typically around sophomore year, to have the necessary knowledge to start applying [S1]." | Relevant | partially accurate |
| 2 | What should a CS internship resume include? | Projects, technical skills, relevant coursework, previous experience if available, and measurable impact. | "A CS internship resume should include skills such as software expertise, programming languages, and other applicable skills [S3]. It's also important to be concise and structure the resume well, providing the information recruiters want [S5]. Additionally, the resume should target the career... and not include irrelevant information [S5]." | Relevant | accurate |
| 3 | Do referrals help compared to cold applications? | Referrals may help get attention, but students still recommend applying broadly and quickly. | "Referrals are implied to be more effective than cold applications... In contrast, cold calling is described as 'spammy'... However, the context does not provide a direct comparison of the two methods' success rates." | Relevant | partially accurate |
| 4 | What should students prepare for online assessments? | Data structures, algorithms, LeetCode-style questions, time management, and practice with coding platforms. | "I don't have enough information on that." | Partially relevant | inaccurate |
| 5 | What advice do students give for behavioral interviews? | Prepare clear stories using a structure like STAR, explain teamwork/conflict examples, and avoid sounding overly scripted. | "I don't have enough information on that." | Partially relevant | inaccurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**
What should students prepare for online assessments?

**What the system returned:**
"I don't have enough information on that."

**Root cause (tied to a specific pipeline stage):**
The failure is mainly in retrieval-to-generation handoff policy, not total absence of relevant corpus content.
The system currently applies a strict distance threshold (0.5). For this query, retrieved chunks had distances around 0.55-0.60, so the model was forced into refusal mode.
This creates a false negative: evidence exists but is rejected by threshold gating.

**What you would change to fix it:**
Use adaptive thresholding (for example, allow slightly higher cutoff for low-margin queries), and/or rerank top chunks with a lightweight cross-encoder before refusal.
I would also tune chunk cleaning and query expansion to improve similarity scores for OA-related language.

---

## Spec Reflection

**One way the spec helped you during implementation:**
The planning spec narrowed implementation choices early: fixed embedding model, chunk-size target, overlap target, and explicit top-k retrieval behavior.
That made Milestones 4 and 5 more objective because I could test against concrete thresholds and expected query types instead of vague "good answer" criteria.

**One way your implementation diverged from the spec, and why:**
The spec expected top-k retrieval to flow into generation directly, but I added a confidence gate that can refuse answers when retrieval distances are weak.
This divergence improved grounding safety, but it also introduced false negatives on borderline-relevant queries, which is now a documented limitation.

---

## AI Usage
**Instance 1**

- *What I gave the AI:*
     Planning constraints (embedding model, top-k retrieval, chunk metadata requirements) and architecture context.
- *What it produced:*
     Milestone 4 embedding/retrieval pipeline code with Chroma integration and CLI commands.
- *What I changed or overrode:*
     I switched to cosine-distance indexing with collection reset support, improved metadata normalization for source attribution, and added evaluation commands for query-based inspection.

**Instance 2**

- *What I gave the AI:*
     Milestone 5 grounding requirements: context-only answering, explicit refusal behavior, and source attribution in output.
- *What it produced:*
     Generation wiring (Groq client, prompt template, ask/chat/serve commands) and a Gradio interface.
- *What I changed or overrode:*
     I hardened grounding by making refusal deterministic for low-confidence retrieval, appended source attribution programmatically, and validated behavior with in-domain and out-of-domain tests.

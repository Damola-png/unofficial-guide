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

## Demo Recording Checklist

Use this order in your video so operation is clear without narration:

1. Show the app launch command:

```bash
python app.py
```

2. Open `http://localhost:7860` and run two in-domain questions:
     - "What should a CS internship resume include?"
     - "Do referrals help compared to cold applications?"

3. Point to both outputs for each question:
     - the grounded answer
     - the "Retrieved from" source panel

4. Run one out-of-domain question:
     - "How do I bake sourdough bread at home?"

5. Show that the system declines with:
     - "I don't have enough information on that."
     - plus retrieved-source transparency.

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.


 ## Possible questions my system should be able to answer, To be removed later 

When should students start applying for CS internships?
What do students recommend putting on a CS internship resume?
Do referrals help more than cold applications?
What should students prepare for online assessments?
What advice do students give for behavioral interviews?
---

## Domain

This system covers unofficial advice for landing CS internships in the US as a college student.
This knowledge is valuable because practical internship tactics are often shared in community sources (student guides, Reddit threads, open-source advice repos), not in official university pages.
Official channels usually provide generic guidance, while this corpus contains concrete strategies about application timing, referral behavior, resume construction, and interview prep.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

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


<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

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

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**
all-MiniLM-L6-v2

**Production tradeoff reflection:**
I chose this model for strong local performance and fast embedding generation with no paid API dependency.
For production with higher budget, I would evaluate a stronger embedding model for better semantic precision on nuanced advice queries, especially where wording differs from source text.
The tradeoff is higher cost/latency versus fewer false negatives and less dependence on aggressive retrieval threshold tuning.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
Generation uses a strict system instruction: answer only from provided retrieved context, do not use outside knowledge, and return "I don't have enough information on that." when evidence is insufficient.
Context is injected as labeled source blocks ([S1], [S2], etc.) with source names and distances.

**How source attribution is surfaced in the response:**
Source attribution is programmatically appended after generation using retrieved metadata (source URL/path, chunk id, rank, distance), so attribution does not depend only on LLM behavior.
The UI also displays a separate "Retrieved from" source panel for transparency.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | When should students start applying for CS internships? | Apply early (late summer/fall) before internship cycle; starting early helps. | Answer said students can apply as early as freshman year, with stronger readiness around sophomore year after DS&A. | Relevant | Partially accurate |
| 2 | What should a CS internship resume include? | Projects, skills, coursework/experience, measurable impact. | Answer included skills, concise structure, career targeting, quantified achievements. | Relevant | Accurate |
| 3 | Do referrals help compared to cold applications? | Referrals can help visibility, but students still apply broadly/quickly. | Answer stated referrals are helpful and cold outreach is viewed less favorably, but noted no direct success-rate comparison in context. | Relevant | Partially accurate |
| 4 | What should students prepare for online assessments? | DS&A, LeetCode-style questions, timing/practice platforms. | System returned: "I don't have enough information on that." due low-confidence retrieval gate. | Partially relevant | Inaccurate |
| 5 | What advice do students give for behavioral interviews? | STAR stories, teamwork/conflict examples, clear communication. | System returned: "I don't have enough information on that." due low-confidence retrieval gate. | Partially relevant | Inaccurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

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

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
The planning spec narrowed implementation choices early: fixed embedding model, chunk-size target, overlap target, and explicit top-k retrieval behavior.
That made Milestones 4 and 5 more objective because I could test against concrete thresholds and expected query types instead of vague "good answer" criteria.

**One way your implementation diverged from the spec, and why:**
The spec expected top-k retrieval to flow into generation directly, but I added a confidence gate that can refuse answers when retrieval distances are weak.
This divergence improved grounding safety, but it also introduced false negatives on borderline-relevant queries, which is now a documented limitation.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

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

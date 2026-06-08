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
python embed_and_retrieve.py serve --top-k 5 --distance-threshold 0.5
```

Grounding behavior:
- The model is prompted to answer only from retrieved chunks.
- If context is insufficient, it is instructed to say so instead of guessing.
- Answers include citation labels like `[S1]`, `[S2]` matching retrieved chunks.

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

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

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

**Overlap:**

**Why these choices fit your documents:**

**Final chunk count:**

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**

**Production tradeoff reflection:**

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

**How source attribution is surfaced in the response:**

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

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

**What the system returned:**

**Root cause (tied to a specific pipeline stage):**

**What you would change to fix it:**

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

**One way your implementation diverged from the spec, and why:**

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
- *What it produced:*
- *What I changed or overrode:*

**Instance 2**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*

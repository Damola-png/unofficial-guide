# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Unofficial guide to getting a Computer Science internship in the US as a college student. This knowledge is extremly useful because studnets share practical advise that could help other peers land high paying computer science internship which is something that a lot of science students are constantly trying to get. 

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

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

---

## Summary of my Domain

My main aim for this project is to focus on unofficial advice on getting a Computer Science internship in the US as a college student. The knowledge is hard to find in official career center pages because students often share the most practical details, such as application timing, referral strategy, online assessments, resume advice, and how many applications it may take, across scattered Reddit threads, blog posts, and peer guides.



## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
For  the chunk size, my plan is to split it around 800 - 1000 characters per chunk. 
**Overlap:**
150- 200 overalp size
**Reasoning:**
A lot of my documents are opinion based, so it would make sense to break them in large chunks to keep some of its content reasonably affective. 
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

all-MiniLM-L6-v2

**Top-k:**
Retrive the tope 5-10 chunks 
**Production tradeoff reflection:**
I would say accuarcy may be the best tradeoff becasue better model may have better accurcay but, it would cost more and take longer time or might even require a paid API calll to be able to work completely well. 

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | When should students start applying for CS internships?| Students often recommend applying early, commonly during late summer or fall before the internship year|
| 2 | What should a CS internship resume include? |Projects, technical skills, relevant coursework, previous experience if available, and measurable impact|
| 3 | Do referrals help compared to cold applications?| Referrals may help get attention, but students still recommend applying broadly and quickly|
| 4 |  What should students prepare for online assessments?|Data structures, algorithms, LeetCode-style questions, time management, and practice with coding platforms|
| 5 | What advice do students give for behavioral interviews?| Prepare clear stories using a structure like STAR, explain teamwork/conflict examples, and avoid sounding overly scripted|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. The biggest issue I could have is if one of the reddit sources have an outdated advice and the users end up using an advice that might not work anymore. 

2. The system may give general advice and not give one specific to the kind of internship (Computer Science) that the user has asked for.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---


![alt text](image.png)

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)

     I plan on using a mixture of Claude and copilot. I will mainly use Claude for larger design and implementataion prompts, and copilot for smaller in-editor suggestions and editting my code. 

     - What you'll give it as input (which sections of this planning.md, which requirements)

     The Documents section of this planning.md
     The project requirement that the pipeline must load raw documents, clean/preprocess them, and produce structured text
     A description of my source types: GitHub guides, Reddit threads, and manually saved text files

     - What you expect it to produce

     A Python script or functions that load documents from a local data folder
     Basic cleaning logic to remove extra whitespace, HTML artifacts, navigation text, and repeated boilerplate
     Metadata for each document, including source title and file path or URL

     - How you'll verify the output matches your spec

     I will print several cleaned documents and compare them to the raw source text
     I will check that the text still contains the useful internship advice
     I will confirm that source metadata is preserved for every document



     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**

"""Generate the canonical 20-query interview-preparation benchmark dataset.

Every query matches the user prompt verbatim (Q01-Q20), with:
- relevant_doc_identifiers
- relevant_chunk_ids
- reference_answer (synthesized directly from corpus chunks)
- expected_sufficiency
- expected_behavior
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "evaluation" / "jev_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = [
    {
        "id": "Q01",
        "query": "What behavioral interview questions are commonly asked at Amazon, and which Leadership Principles do they test?",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "Hybrid retrieval with company metadata filter (Amazon) targeting Leadership Principles behavioral questions.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf",
            "Amazon.pdf",
            "documents/knowledge_base.md"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "63ca9464d9701d956cedf155ad62c3971b8b53ed9e6e3b17d162fe92e627c05d",
            "957b544449c7d3ed073de9886d80012604b77a8dd6bf98ff5cccc5554da987a7",
            "4ffcb1494c286119cc320788a46a1c5d65896d2835bca423b050ae0ffccfbe9d",
            "de09e6c9d70376f7c8e2f5d8ca8813ffb60e57e3b6bcc727d66e7a33971123bc"
        ],
        "reference_answer": (
            "Amazon behavioral interview questions are structured around its 16 Leadership Principles (LPs). "
            "Commonly asked questions include: "
            "1. 'Tell me about a time you had a disagreement with your manager or colleague' (tests 'Have Backbone; Disagree and Commit'). "
            "2. 'Describe a situation where you took on a task outside your formal responsibility' (tests 'Ownership'). "
            "3. 'Give an example of a difficult problem where you had to analyze deep metrics and root causes' (tests 'Dive Deep'). "
            "4. 'Describe a project where you prioritized long-term customer benefit over short-term expediency' (tests 'Customer Obsession'). "
            "5. 'Tell me about a time you delivered a project under tight constraints with limited resources' (tests 'Frugality' and 'Deliver Results'). "
            "Candidates are expected to format all behavioral responses using the STAR method (Situation, Task, Action, Result) with quantified impact."
        )
    },
    {
        "id": "Q02",
        "query": "I'm preparing for an Amazon software engineering interview. Give me behavioral questions I should practice, and explain what a strong answer should demonstrate for each.",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "Dense semantic retrieval with multi-document synthesis over Amazon SDE behavioral guidance.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "63ca9464d9701d956cedf155ad62c3971b8b53ed9e6e3b17d162fe92e627c05d",
            "4ffcb1494c286119cc320788a46a1c5d65896d2835bca423b050ae0ffccfbe9d"
        ],
        "reference_answer": (
            "For an Amazon SDE interview, strong practice questions and what an effective answer must demonstrate include: "
            "1. Disagreement / Conflict: 'Tell me about a time you disagreed with a colleague or manager.' Strong answers demonstrate 'Have Backbone; Disagree and Commit'—explaining the technical reasoning with data, maintaining professional respect, and fully committing once a team decision was finalized. "
            "2. Failure / Learning: 'Describe a project that failed or missed a deadline.' Strong answers demonstrate 'Earn Trust' and self-reflection—owning the mistake without deflecting blame, identifying root causes ('Dive Deep'), and showing preventive mechanisms implemented afterward. "
            "3. Ownership: 'Give an example of when you took initiative beyond your job description.' Strong answers demonstrate long-term thinking over 'that's not my job', showing end-to-end accountability. "
            "All answers should strictly follow the STAR method, emphasizing specific actions taken by the candidate ('I' rather than vague 'we') and measurable outcomes."
        )
    },
    {
        "id": "Q03",
        "query": "What technical topics should I prepare for a Google software engineering interview, particularly around data structures and algorithms?",
        "category": "Technical",
        "difficulty": "Medium",
        "expected_behavior": "Hybrid BM25 + dense retrieval filtering on Google interview handbooks and DSA guides.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Google",
        "relevant_doc_identifiers": [
            "Google Interview Guide.pdf",
            "Google.pdf",
            "Top google question part-1.pdf"
        ],
        "relevant_chunk_ids": [
            "570cfeea8e2118adc7c88231fa008c5b08649a93f1cc2c10d176d7d6c20ce407",
            "195ea806dc0cb627bf9c14828b09b2976991b04b344b7be9de5f92d8e660ea64",
            "806e40d67ffae54b9d18c54f10b54db9dfe71841f810b05d49dde14b4f9483cc",
            "5e78e8a387d3a5eedc168ddfbd0df7e158ff33074f2d50aa5a82d3a48b33fd74"
        ],
        "reference_answer": (
            "For a Google software engineering interview, technical preparation should prioritize: "
            "1. Core Data Structures: Arrays, Linked Lists, Stacks, Queues, Hash Tables, Binary Trees, Binary Search Trees (BST), Heaps/Priority Queues, and Trie structures. "
            "2. Graph Algorithms: Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's algorithm, Topological Sort, and Union-Find (Disjoint Set). "
            "3. Algorithmic Techniques: Binary Search, Two Pointers, Sliding Window, Recursion & Backtracking, Dynamic Programming (1D and 2D), and Greedy approaches. "
            "4. Complexity Analysis: Rigorous asymptotic Big-O time and space complexity analysis, including trade-offs between iterative and recursive solutions."
        )
    },
    {
        "id": "Q04",
        "query": "For a Google software engineering interview, what kinds of coding problems and problem-solving skills should I expect across the technical rounds?",
        "category": "Technical",
        "difficulty": "Medium",
        "expected_behavior": "Dense semantic retrieval synthesizing problem-solving rubrics and technical rounds from Google Interview Guide.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Google",
        "relevant_doc_identifiers": [
            "Google Interview Guide.pdf",
            "Google.pdf"
        ],
        "relevant_chunk_ids": [
            "806e40d67ffae54b9d18c54f10b54db9dfe71841f810b05d49dde14b4f9483cc",
            "570cfeea8e2118adc7c88231fa008c5b08649a93f1cc2c10d176d7d6c20ce407",
            "195ea806dc0cb627bf9c14828b09b2976991b04b344b7be9de5f92d8e660ea64"
        ],
        "reference_answer": (
            "Google's technical rounds assess four key competencies: Coding Skills, Analytical Thinking, Computer Science Fundamentals, and Communication. "
            "Candidates should expect algorithmic problem solving on Google Docs or whiteboards without auto-complete or compiler assistance. "
            "Problem types typically involve LeetCode Medium to Hard algorithmic challenges requiring candidates to: "
            "1. Clarify constraints and edge cases before coding. "
            "2. Propose a baseline approach and evaluate time/space complexity before writing code. "
            "3. Write clean, idiomatic, bug-free code with proper naming conventions and modularity. "
            "4. Dry-run test cases and identify boundary conditions (e.g., null pointers, empty arrays, integer overflows)."
        )
    },
    {
        "id": "Q05",
        "query": "Compare Amazon and Google software engineering interviews in terms of their behavioral and technical interview expectations.",
        "category": "Multi-Company",
        "difficulty": "Hard",
        "expected_behavior": "Multi-hop cross-company retrieval synthesizing Amazon Leadership Principles vs. Google coding/Googliness rubrics.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": None,
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Google Interview Guide.pdf",
            "Amazon.pdf",
            "Google.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "806e40d67ffae54b9d18c54f10b54db9dfe71841f810b05d49dde14b4f9483cc",
            "570cfeea8e2118adc7c88231fa008c5b08649a93f1cc2c10d176d7d6c20ce407",
            "b373da1123f0d4141393889a2fb10db8cf3d812844227d363b9e293d861ee324"
        ],
        "reference_answer": (
            "Amazon and Google software engineering interviews differ fundamentally in their evaluation emphasis: "
            "1. Behavioral Emphasis: Amazon places approximately 50% of hiring weight on behavioral fit evaluated against its 16 Leadership Principles (LPs), including a dedicated Bar Raiser round. Google focuses primarily on 'Googliness & Leadership' (intellectual humility, navigating ambiguity, collaborative problem solving), with behavioral assessment integrated into standard rounds. "
            "2. Technical Expectations: Google emphasizes mathematical rigor, complex algorithmic design, dynamic programming, and clean syntax in language-neutral problem solving. Amazon evaluates practical DSA problem-solving and software design with heavy emphasis on object-oriented programming (OOP), system scalability, and pragmatic implementation trade-offs."
        )
    },
    {
        "id": "Q06",
        "query": "What does Amazon's Leadership Principle \"Dive Deep\" mean in an interview context, and what type of behavioral question could be used to evaluate it?",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "Semantic retrieval and HyDE expansion on Amazon Leadership Principle 'Dive Deep'.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon.pdf",
            "Amazon Part 2.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "b373da1123f0d4141393889a2fb10db8cf3d812844227d363b9e293d861ee324",
            "4ffcb1494c286119cc320788a46a1c5d65896d2835bca423b050ae0ffccfbe9d"
        ],
        "reference_answer": (
            "In Amazon's interview context, 'Dive Deep' means leaders operate at all levels, stay connected to the details, audit frequently, and remain skeptical when metrics and anecdotes differ. No task is beneath them. "
            "Typical behavioral interview questions evaluating 'Dive Deep' include: "
            "- 'Tell me about a complex technical problem where you had to dig into the details to find the root cause.' "
            "- 'Describe a time when you discovered an anomaly in data or metrics and investigated beyond the surface explanation.' "
            "A strong answer demonstrates drilling down through multiple layers of symptoms to identify root causes using data, log inspection, or code profiling."
        )
    },
    {
        "id": "Q07",
        "query": "I keep seeing questions about ownership in Amazon interviews. What interview questions are associated with this theme, and what evidence from Amazon's interview guidance supports that?",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "Query rewrite and semantic-lexical retrieval targeting Amazon 'Ownership' principle.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf"
        ],
        "relevant_chunk_ids": [
            "63ca9464d9701d956cedf155ad62c3971b8b53ed9e6e3b17d162fe92e627c05d",
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "4ffcb1494c286119cc320788a46a1c5d65896d2835bca423b050ae0ffccfbe9d"
        ],
        "reference_answer": (
            "Amazon defines 'Ownership' as thinking long term and not sacrificing long-term value for short-term results. Leaders act on behalf of the entire company, beyond just their own team, and never say 'that's not my job.' "
            "Common interview questions associated with Ownership include: "
            "1. 'Tell me about a time you took on a task or project that was outside your defined responsibilities.' "
            "2. 'Describe a situation where you noticed an issue that wasn't being addressed by anyone and stepped up to fix it.' "
            "Amazon's interview guidance emphasizes that candidates must demonstrate personal accountability and explain the long-term benefit of their initiative."
        )
    },
    {
        "id": "Q08",
        "query": "Suppose I am asked a behavioral question about a project that failed during an Amazon interview. Which Leadership Principles could be relevant, and how should I structure my response?",
        "category": "Multi-Hop",
        "difficulty": "Hard",
        "expected_behavior": "Multi-hop behavioral synthesis linking failure scenarios to Amazon LPs (Earn Trust, Have Backbone, Dive Deep).",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "4ffcb1494c286119cc320788a46a1c5d65896d2835bca423b050ae0ffccfbe9d",
            "63ca9464d9701d956cedf155ad62c3971b8b53ed9e6e3b17d162fe92e627c05d"
        ],
        "reference_answer": (
            "When responding to a project failure question in an Amazon interview, multiple Leadership Principles are evaluated: "
            "1. 'Earn Trust' (being vocally self-critical and owning the failure without blaming colleagues or tools). "
            "2. 'Dive Deep' (identifying the specific technical or operational root cause rather than settling for high-level excuses). "
            "3. 'Are Right, A Lot' (demonstrating how decision-making evolved as new information came to light). "
            "The response should be structured strictly using the STAR format: "
            "- Situation/Task: Contextualize the ambitious objective. "
            "- Action: Highlight what went wrong, how you intervened, and how you communicated transparently. "
            "- Result/Learning: Emphasize the concrete lessons learned and what automated checks, guardrails, or architectural changes you enacted to ensure the failure never happens again."
        )
    },
    {
        "id": "Q09",
        "query": "What are the main interview rounds for an Amazon software engineering candidate, and what should I prepare for in each round?",
        "category": "Interview Process",
        "difficulty": "Medium",
        "expected_behavior": "Process retrieval with metadata filtering on Amazon interview structure.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf",
            "Amazon.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "957b544449c7d3ed073de9886d80012604b77a8dd6bf98ff5cccc5554da987a7",
            "b373da1123f0d4141393889a2fb10db8cf3d812844227d363b9e293d861ee324"
        ],
        "reference_answer": (
            "Amazon's SDE hiring process consists of three main stages: "
            "1. Online Assessment (OA): Typically 2 coding questions (debugging/DSA) plus a Work Style Assessment reflecting Amazon Leadership Principles. "
            "2. Technical Phone Screen: 45-60 minutes covering 1 coding problem (data structures/algorithms) and 1-2 behavioral LP questions. "
            "3. Onsite / Final Loop: 4-5 interviews (60 mins each) including: "
            "   - Coding/DSA (2 rounds): In-depth algorithmic problem solving. "
            "   - System Design / OOD (1 round): Low-level object-oriented design (for SDE I) or high-level distributed systems design (for SDE II). "
            "   - Bar Raiser (1 round): Independent interviewer evaluating candidate fit against Amazon's bar, heavily weighted on Leadership Principles. "
            "Every single interview round in the loop contains 15-20 minutes of behavioral LP questions."
        )
    },
    {
        "id": "Q10",
        "query": "I'm interviewing for an AI/ML role at NVIDIA. What technical areas and interview topics should I prioritize?",
        "category": "Technical",
        "difficulty": "Medium",
        "expected_behavior": "Dense semantic retrieval with role-specific extraction over NVIDIA handbook.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "NVIDIA",
        "relevant_doc_identifiers": [
            "NVIDIA.pdf"
        ],
        "relevant_chunk_ids": [
            "beb48e6261f51fd37365168e6ea74caf149574db4c34aea0eccbe00a781b6417"
        ],
        "reference_answer": (
            "For an AI/ML engineering role at NVIDIA, candidates should prioritize five core technical areas: "
            "1. Parallel Computing & GPU Architecture: CUDA programming, thread hierarchy (grids, blocks, warps), shared memory vs. global memory bandwidth, and kernel optimization. "
            "2. Deep Learning Frameworks: PyTorch, TensorFlow, and inference acceleration using NVIDIA TensorRT. "
            "3. Generative AI & LLM Systems: Designing scalable distributed model training pipelines and high-throughput low-latency inference engines. "
            "4. Systems & Architecture: CPU vs. GPU architecture, cache hierarchy, memory management, and multi-threading/synchronization primitives in C/C++. "
            "5. Core DSA: Graph algorithms (Dijkstra, Topological Sort), bit manipulation, and arrays/strings optimized for performance."
        )
    },
    {
        "id": "Q11",
        "query": "What kinds of GPU, CUDA, or deep-learning questions might be relevant when preparing for an NVIDIA technical interview?",
        "category": "Technical",
        "difficulty": "Medium",
        "expected_behavior": "Lexical BM25 and exact terminology reranking for GPU/CUDA/Deep Learning.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "NVIDIA",
        "relevant_doc_identifiers": [
            "NVIDIA.pdf"
        ],
        "relevant_chunk_ids": [
            "beb48e6261f51fd37365168e6ea74caf149574db4c34aea0eccbe00a781b6417"
        ],
        "reference_answer": (
            "Relevant GPU, CUDA, and deep-learning questions for NVIDIA include: "
            "1. GPU Architecture & Memory: Explain the difference between shared memory and global memory in CUDA. How do you prevent warp divergence? What is memory coalescing? "
            "2. CUDA Programming: Write or optimize a basic vector addition or matrix multiplication kernel in CUDA. How do you handle synchronization with __syncthreads()? "
            "3. Deep Learning Acceleration: How does TensorRT optimize neural network execution (layer fusion, precision calibration like FP16/INT8)? "
            "4. System Design for AI: Design a scalable distributed model training pipeline or real-time low-latency object detection system."
        )
    },
    {
        "id": "Q12",
        "query": "What behavioral interview topics should I prepare for at NVIDIA, and how do they differ from the technical topics I should study?",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "Hybrid retrieval separating behavioral vs. technical interview rubrics for NVIDIA.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "NVIDIA",
        "relevant_doc_identifiers": [
            "NVIDIA.pdf"
        ],
        "relevant_chunk_ids": [
            "beb48e6261f51fd37365168e6ea74caf149574db4c34aea0eccbe00a781b6417"
        ],
        "reference_answer": (
            "NVIDIA behavioral topics focus on cultural fit, innovation drive, and problem solving under constraints: "
            "1. 'Why NVIDIA?' (demonstrating passion for GPU acceleration, gaming, AI, or autonomous systems). "
            "2. Performance Optimization: 'Tell me about a time you had to optimize code for performance.' "
            "3. Rapid Learning: 'Describe a project where you had to learn a complex technology quickly.' "
            "4. Conflict Resolution: 'How do you handle disagreements in a technical discussion?' "
            "This differs sharply from NVIDIA's technical rounds, which focus on low-level C++ mechanics, memory bandwidth, CUDA kernels, and deep learning framework optimizations."
        )
    },
    {
        "id": "Q13",
        "query": "If an interviewer asks me to describe a time I disagreed with a teammate, what company-specific interview expectations should I consider when answering for Amazon?",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "HyDE and semantic retrieval targeting Amazon's 'Disagree and Commit' principle.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf",
            "documents/knowledge_base.md"
        ],
        "relevant_chunk_ids": [
            "63ca9464d9701d956cedf155ad62c3971b8b53ed9e6e3b17d162fe92e627c05d",
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "de09e6c9d70376f7c8e2f5d8ca8813ffb60e57e3b6bcc727d66e7a33971123bc"
        ],
        "reference_answer": (
            "When answering a disagreement question for Amazon, candidates must align directly with the Leadership Principle 'Have Backbone; Disagree and Commit': "
            "1. Leaders are obligated to respectfully challenge decisions when they disagree, even when doing so is uncomfortable or exhausting. They do not compromise for the sake of social cohesion. "
            "2. The disagreement must be grounded in data, customer impact, or architectural trade-offs, rather than personal preference. "
            "3. Once a decision is determined, leaders commit wholly to the outcome. "
            "The response must show professional advocacy during deliberation followed by enthusiastic execution once the final call was made."
        )
    },
    {
        "id": "Q14",
        "query": "Which Amazon interview questions would require me to discuss measurable impact, and what Leadership Principles are most closely connected to them?",
        "category": "Behavioral",
        "difficulty": "Medium",
        "expected_behavior": "Semantic reranking connecting behavioral questions with 'Deliver Results' and quantified impact.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": "Amazon",
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Amazon Part 2.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "4ffcb1494c286119cc320788a46a1c5d65896d2835bca423b050ae0ffccfbe9d"
        ],
        "reference_answer": (
            "Amazon interview questions requiring measurable impact relate primarily to 'Deliver Results' and 'Customer Obsession': "
            "1. 'Tell me about a time you delivered an important project on time despite tight constraints.' "
            "2. 'Describe a situation where you improved system performance or efficiency.' "
            "3. 'Give an example of a time you exceeded customer expectations.' "
            "Amazon explicitly expects metrics in the 'Result' phase of the STAR method—such as percentage latency reductions, cost savings in dollars, uptime improvements (e.g., 99.99%), or user adoption growth numbers."
        )
    },
    {
        "id": "Q15",
        "query": "I'm preparing for interviews at multiple companies. How should my preparation differ between Amazon's behavioral interviews and NVIDIA's technical interviews?",
        "category": "Multi-Company",
        "difficulty": "Hard",
        "expected_behavior": "Multi-hop cross-company synthesis contrasting Amazon LP focus with NVIDIA C++/CUDA hardware-aware depth.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": None,
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "NVIDIA.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "beb48e6261f51fd37365168e6ea74caf149574db4c34aea0eccbe00a781b6417"
        ],
        "reference_answer": (
            "Preparation differs substantially across these two targets: "
            "1. Amazon Behavioral Preparation: Focus on assembling 6-8 comprehensive STAR stories demonstrating Amazon's 16 Leadership Principles (Customer Obsession, Ownership, Dive Deep, Deliver Results). Prepare to be pressed for granular details and quantifiable results across multiple rounds. "
            "2. NVIDIA Technical Preparation: Focus on low-level systems engineering and hardware acceleration: master modern C++, understand parallel programming in CUDA, review memory bandwidth/cache hierarchies, and prepare for domain-specific questions in TensorRT, GPU architecture, or AI system design."
        )
    },
    {
        "id": "Q16",
        "query": "What interview questions should I practice if I want to demonstrate problem solving, technical depth, and ownership rather than just memorizing coding patterns?",
        "category": "Semantic",
        "difficulty": "Hard",
        "expected_behavior": "Semantic retrieval synthesizing system design, trade-off analysis, and complex behavioral problems.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": None,
        "relevant_doc_identifiers": [
            "Google Interview Guide.pdf",
            "How to prepare for your SDE interview at Amazon.pdf",
            "documents/knowledge_base.md"
        ],
        "relevant_chunk_ids": [
            "570cfeea8e2118adc7c88231fa008c5b08649a93f1cc2c10d176d7d6c20ce407",
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "06bbe57de2754fd59b09662d918a034e7d81cf608a126ebd33b5f969ddc1c9ac"
        ],
        "reference_answer": (
            "To demonstrate genuine problem-solving depth and ownership rather than rote pattern memorization: "
            "1. System Design Questions (e.g., 'Design a URL shortener like bit.ly', 'Design a rate limiter', 'Design a model inference pipeline'): Tests data modeling, scalability trade-offs, consistency vs. availability, and capacity planning. "
            "2. Algorithmic Trade-off Challenges: Questions where initial solutions have high complexity and require custom data structure design (e.g., LRU Cache design combining a Hash Map with a Doubly Linked List for O(1) operations). "
            "3. Engineering Ownership Questions: 'Describe an architecture or production bug you investigated and resolved end-to-end', highlighting root cause analysis and proactive mitigation."
        )
    },
    {
        "id": "Q17",
        "query": "Give me examples of technical interview questions where the exact terminology matters, and explain what concepts each question is testing.",
        "category": "Lexical",
        "difficulty": "Medium",
        "expected_behavior": "Lexical precision retrieval on exact technical interview terminology.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": None,
        "relevant_doc_identifiers": [
            "README.pdf",
            "NVIDIA.pdf",
            "documents/knowledge_base.md"
        ],
        "relevant_chunk_ids": [
            "21104233549d2211e62effbe8268b08f34f641b2a7bfc3090dde444637dfee3d",
            "beb48e6261f51fd37365168e6ea74caf149574db4c34aea0eccbe00a781b6417",
            "f8a827c7b35d7ac000749c8c4a04f46f9a72c5beec8e9ee23de5dc0330d268ec"
        ],
        "reference_answer": (
            "Examples where exact technical terminology is critical include: "
            "1. 'Concurrency vs. Parallelism': Concurrency is about dealing with lots of things at once (structure); parallelism is about doing lots of things at once (execution across cores/GPUs). "
            "2. 'ACID vs. BASE Properties': Atomicity, Consistency, Isolation, Durability in relational stores vs. Basically Available, Soft state, Eventual consistency in distributed NoSQL stores. "
            "3. 'Shared Memory vs. Global Memory' in CUDA: Shared memory resides on-chip (extremely high bandwidth, low latency, scoped to a thread block) whereas global memory resides in device DRAM (high latency, accessible across all blocks)."
        )
    },
    {
        "id": "Q18",
        "query": "If the available interview material does not contain enough evidence to answer a company-specific question, how should the system respond instead of making up an answer?",
        "category": "Insufficient Evidence",
        "difficulty": "Hard",
        "expected_behavior": "Evidence sufficiency evaluation detecting lack of grounding; strict abstention notice.",
        "expected_sufficiency": "INSUFFICIENT",
        "should_abstain": True,
        "target_company": "UNSUPPORTED_META",
        "relevant_doc_identifiers": [],
        "relevant_chunk_ids": [],
        "reference_answer": (
            "When the available knowledge corpus does not contain verified interview records or guidance for a queried topic or company, "
            "the system must strictly abstain from generating ungrounded advice. It must return an explicit Evidence Sufficiency Notice "
            "explaining that the knowledge corpus does not contain verified records for the target domain, and recommend consulting supported companies."
        )
    },
    {
        "id": "Q19",
        "query": "What is the difference between preparing for behavioral questions based on company principles and preparing for technical coding questions? Use the interview material in the knowledge base to explain.",
        "category": "Interview Process",
        "difficulty": "Medium",
        "expected_behavior": "Hybrid retrieval and multi-document synthesis over behavioral vs. technical handbooks.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": None,
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Google Interview Guide.pdf",
            "README.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "570cfeea8e2118adc7c88231fa008c5b08649a93f1cc2c10d176d7d6c20ce407",
            "21104233549d2211e62effbe8268b08f34f641b2a7bfc3090dde444637dfee3d"
        ],
        "reference_answer": (
            "Based on the interview handbooks in the knowledge base: "
            "1. Behavioral Preparation (Principle-Based): Requires extracting past projects into structured STAR narratives (Situation, Task, Action, Result) demonstrating specific core values (e.g., Amazon's Leadership Principles or Google's collaborative Googliness). Evaluation focuses on interpersonal conflict, technical ownership, handling failure, and customer advocacy. "
            "2. Technical Coding Preparation: Focuses on algorithmic problem solving, time/space complexity analysis (Big-O), choosing optimal data structures, and writing clean, syntax-error-free code under time pressure."
        )
    },
    {
        "id": "Q20",
        "query": "I have an upcoming software engineering interview and only one week to prepare. Based on the company's interview material in the knowledge base, what should I prioritize across behavioral, technical, and interview-process preparation?",
        "category": "Multi-Hop",
        "difficulty": "Hard",
        "expected_behavior": "Complex agentic multi-hop retrieval synthesizing time-constrained preparation strategies.",
        "expected_sufficiency": "SUFFICIENT",
        "should_abstain": False,
        "target_company": None,
        "relevant_doc_identifiers": [
            "How to prepare for your SDE interview at Amazon.pdf",
            "Google Interview Guide.pdf",
            "README.pdf"
        ],
        "relevant_chunk_ids": [
            "3c236f46c2b8d19fbdbc209747fbdc6e9c00aa617f81203465e5d4d44e9cb75b",
            "570cfeea8e2118adc7c88231fa008c5b08649a93f1cc2c10d176d7d6c20ce407",
            "21104233549d2211e62effbe8268b08f34f641b2a7bfc3090dde444637dfee3d"
        ],
        "reference_answer": (
            "With one week to prepare, candidate guidance from the knowledge base recommends a 3-part triage: "
            "1. Technical Coding (50% of time): Focus on high-frequency patterns—Two Pointers, Sliding Window, Trees/BFS/DFS, Hash Tables. Review Big-O complexity and practice writing code on a plain text editor or whiteboard without auto-complete. "
            "2. Behavioral Preparation (30% of time): Outline 5-6 versatile STAR stories covering high-priority principles (Ownership, Dealing with Conflict, Overcoming Failure, Measurable Results) that can be adapted to multiple questions. "
            "3. Interview Process Familiarization (20% of time): Understand the company's specific loop structure (e.g., Amazon Bar Raiser expectations vs. Google coding rubrics) and prepare thoughtful questions to ask the interviewers."
        )
    },
]


def main():
    out_file = OUT_DIR / "benchmark_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(DATASET, f, indent=2)
    print(f"Generated {len(DATASET)} canonical benchmark test cases to {out_file}")


if __name__ == "__main__":
    main()

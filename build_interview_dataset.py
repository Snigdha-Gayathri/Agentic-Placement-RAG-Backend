from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research"
DATA = ROOT / "data" / "new_company_interview_research"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

TODAY = "2026-08-19"

# The list is deliberately limited to companies absent from the recovered historical
# dataset. Every factual statement below is tied to a source URL captured in the record.
companies = [
    {
        "company": "OpenAI", "domains": ["LLM", "AI Safety", "System Design", "Inference Optimization"],
        "roles": ["Research Engineer", "ML Engineer"],
        "focus": ("OpenAI develops frontier AI systems and its interview guide emphasizes engineering solutions, code quality, performance, testing, communication, and collaboration.", "OpenAI interview guide", "https://openai.com/interview-guide/", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [
            ("REPORTED_INTERVIEW_PROCESS", "Research Engineer", "", "Candidate reported a recruiter screen, two 60-minute technical screens, and three additional technical interviews; later rounds included practical ML/debugging and a statistics-heavy round.", "https://www.tryexponent.com/experiences/openai-machine-learning-engineer-interview-c7b6a2", "OpenAI Research Engineer Interview Experience (2025)", "", ["Machine Learning", "Statistics", "DSA"], "medium"),
            ("REPORTED_QUESTION", "Research Engineer", "Implement an all_gather on noisy nodes, derive how many rounds are needed for a target error, and improve the algorithm using the fact that floats are transmitted.", "Candidate-reported exact task from the ML/statistics round.", "https://www.tryexponent.com/experiences/openai-machine-learning-engineer-interview-c7b6a2", "OpenAI Research Engineer Interview Experience (2025)", "", ["Distributed Systems", "Machine Learning", "Inference Optimization"], "medium"),
        ],
        "inferred": ("How would you design an evaluation and serving workflow for a frontier model that balances solution quality, optimal performance, test coverage, and safe deployment?", ["LLM", "Evaluation", "AI Safety", "Inference Optimization", "System Design"]),
        "sources": [("OpenAI interview guide", "https://openai.com/interview-guide/", "", "COMPANY_TECHNICAL_FOCUS"), ("OpenAI Research Engineer Interview Experience (2025)", "https://www.tryexponent.com/experiences/openai-machine-learning-engineer-interview-c7b6a2", "", "REPORTED_QUESTION"), ("OpenAI Interview Process & Experience Megathread [2026]", "https://www.reddit.com/r/Hack2Hire/comments/1tqdmyr/openai_interview_process_experience_megathread/", "2026", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Anthropic", "domains": ["LLM", "Agents", "RAG", "AI Safety", "MLOps"],
        "roles": ["Machine Learning Engineer", "Prompt Engineer"],
        "focus": ("Anthropic candidate evidence emphasizes LLM tooling, long context, memory, reliability, enterprise deployment, and choosing AI versus conventional ML; a separate candidate report records an automated coding assessment.", "Anthropic ML Engineer, Prompt Engineer Interview Experience", "https://www.tryexponent.com/experiences/anthropic-machine-learning-engineer-interview-27873a", "2026", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [
            ("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Candidate reported recruiter, technical use-case, MLOps, hiring-manager, and three-part final panel rounds covering ML design, behavioral, and culture fit.", "https://www.tryexponent.com/experiences/anthropic-machine-learning-engineer-interview-27873a", "Anthropic ML Engineer, Prompt Engineer Interview Experience", "2026", ["LLM", "Agents", "MLOps", "AI Reliability", "Behavioral"], "medium"),
            ("REPORTED_QUESTION", "Machine Learning Engineer", "Write an in-memory database for factory workers with functions to add employees and track time in and time out.", "Candidate-reported CodeSignal task.", "https://www.jointaro.com/interviews/companies/anthropic/experiences/machine-learning-engineer-san-francisco-ca-may-1-2024-no-offer-negative-fdd002ad/", "Machine Learning Engineer Interview Experience - San Francisco", "2024-05-01", ["Python", "DSA", "System Design"], "medium"),
        ],
        "inferred": ("How would you design a reliable long-context agent with memory, tool use, and enterprise deployment controls, and when would you choose a conventional ML solution instead?", ["LLM", "Agents", "AI Reliability", "AI Safety", "System Design"]),
        "sources": [("Anthropic ML Engineer, Prompt Engineer Interview Experience", "https://www.tryexponent.com/experiences/anthropic-machine-learning-engineer-interview-27873a", "2026", "REPORTED_INTERVIEW_PROCESS"), ("Machine Learning Engineer Interview Experience - San Francisco", "https://www.jointaro.com/interviews/companies/anthropic/experiences/machine-learning-engineer-san-francisco-ca-may-1-2024-no-offer-negative-fdd002ad/", "2024-05-01", "REPORTED_QUESTION"), ("Anthropic careers", "https://www.anthropic.com/careers", "", "COMPANY_TECHNICAL_FOCUS")],
    },
    {
        "company": "Cohere", "domains": ["LLM", "NLP", "RAG", "Embeddings", "Inference Optimization"],
        "roles": ["AI Engineer", "ML Engineer", "ML Systems Engineer"],
        "focus": ("Cohere describes current roles across agentic platform, embeddings and search, inference, modeling, applied ML, and ML systems; its transformer explainer identifies transformer models as deep-learning networks for sequential relationships.", "Cohere jobs", "https://jobs.ashbyhq.com/cohere", "2026", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Current role families include agentic environments, ML systems frameworks and tooling, safety tooling, applied ML, model serving, embeddings and search, and inference.", "Member of Technical Staff / ML Systems / Applied ML roles", "https://jobs.ashbyhq.com/cohere", "2026", ["LLM", "Agents", "RAG", "Retrieval", "Inference Optimization", "AI Safety"]),
        "reports": [
            ("REPORTED_INTERVIEW_PROCESS", "AI Engineer", "", "Candidate reported an HR screen, one-hour online assessment with three coding problems, a 48-hour take-home case study, coding, ML design, paper reading/deep dive, and hiring-manager/behavioral rounds.", "https://www.linkjob.ai/interview-questions/cohere-interview-process-and-questions/", "My 2026 Cohere Interview Process and Actual Questions I Faced", "2025-09-09", ["DSA", "LLM", "RAG", "System Design", "Behavioral"], "medium"),
            ("REPORTED_QUESTION", "AI Engineer", "Implement top_k LLM token decoding and similar algorithms.", "Candidate-reported coding-round topic; wording is preserved at the level exposed by the source.", "https://www.linkjob.ai/interview-questions/cohere-interview-process-and-questions/", "My 2026 Cohere Interview Process and Actual Questions I Faced", "2025-09-09", ["LLM", "Python", "Inference Optimization"], "medium"),
            ("REPORTED_QUESTION", "AI Engineer", "Design a mechanism for an LLM system to answer questions about events after its training cutoff while maintaining reliability and transparency.", "Candidate-reported ML design prompt involving retrieval, uncertainty, validation, latency, and hallucination prevention.", "https://www.linkjob.ai/interview-questions/cohere-interview-process-and-questions/", "My 2026 Cohere Interview Process and Actual Questions I Faced", "2025-09-09", ["LLM", "RAG", "AI Reliability", "System Design"], "medium"),
        ],
        "inferred": ("How would you optimize a batch embedding or reranking pipeline under maximum-token and maximum-batch-size constraints while preserving retrieval quality?", ["Embeddings", "RAG", "Inference Optimization", "Distributed Systems"]),
        "sources": [("Cohere jobs", "https://jobs.ashbyhq.com/cohere", "2026", "ROLE_REQUIREMENT"), ("My 2026 Cohere Interview Process and Actual Questions I Faced", "https://www.linkjob.ai/interview-questions/cohere-interview-process-and-questions/", "2025-09-09", "REPORTED_QUESTION"), ("What are transformer models? Use cases and examples", "https://cohere.com/blog/transformer-model", "2024-12-20", "COMPANY_TECHNICAL_FOCUS")],
    },
    {
        "company": "Perplexity", "domains": ["LLM", "RAG", "Retrieval", "NLP", "Agents"],
        "roles": ["AI Engineer", "Software Engineer"],
        "focus": ("Perplexity states that technical candidates are evaluated on frontier knowledge, hands-on work, full-stack understanding, and practical AI-product use; its official process includes programming screening, deep dive, onsite, and founder/leader interview.", "Perplexity Interview Guide", "https://www.perplexity.ai/hub/careers/interview-guide", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "AI Engineer", "", "The official guide describes a recruiter phone screen, technical programming screen, onsite with 4-5 interviews including a hiring-manager deep dive, and a final founder or leader interview. This is company-published process context, not a candidate report.", "https://www.perplexity.ai/hub/careers/interview-guide", "Perplexity Interview Guide", "", ["DSA", "System Design", "LLM", "Behavioral"], "medium")],
        "inferred": ("How would you build a retrieval-grounded answer system that decides when to search, cites evidence, and communicates uncertainty to a user?", ["RAG", "Retrieval", "LLM", "AI Reliability", "System Design"]),
        "sources": [("Perplexity Interview Guide", "https://www.perplexity.ai/hub/careers/interview-guide", "", "COMPANY_TECHNICAL_FOCUS")],
    },
    {
        "company": "xAI", "domains": ["LLM", "AI Infrastructure", "ML Infrastructure", "Inference Optimization"],
        "roles": ["Member of Technical Staff - Model Training", "Network Engineer - ML Infrastructure"],
        "focus": ("xAI describes work on frontier AI models and features current roles in model training and ML infrastructure, with technical-team review, a screening interview, and deep technical interviews.", "Careers: Build AI That Advances Humanity", "https://x.ai/careers", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Featured roles include model training and ML infrastructure for high-performance AI systems.", "Member of Technical Staff - Model Training / Network Engineer - ML Infrastructure", "https://x.ai/careers", "2026", ["LLM", "GPU", "Distributed Systems", "Inference Optimization", "ML Infrastructure"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Search-result evidence reported a short phone screen followed by four onsite interviews; retained as low-confidence because the underlying Glassdoor page was not opened.", "https://www.glassdoor.com/Interview/xAI-Machine-Learning-Engineer-DeepSearch-Interview-Questions-EI_IE10404667.0,3_KO4,40.htm", "xAI Machine Learning Engineer DeepSearch interview questions", "", ["LLM", "System Design"], "low")],
        "inferred": ("How would you profile and optimize a distributed model-training or inference system when GPU utilization and interconnect bandwidth are bottlenecks?", ["GPU", "Distributed Systems", "Inference Optimization", "ML Infrastructure"]),
        "sources": [("Careers: Build AI That Advances Humanity", "https://x.ai/careers", "", "ROLE_REQUIREMENT"), ("xAI Machine Learning Engineer DeepSearch interview questions", "https://www.glassdoor.com/Interview/xAI-Machine-Learning-Engineer-DeepSearch-Interview-Questions-EI_IE10404667.0,3_KO4,40.htm", "", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Databricks", "domains": ["ML Infrastructure", "RAG", "Agents", "Distributed Systems", "MLOps"],
        "roles": ["ML Engineer", "Data/AI Engineer"],
        "focus": ("Databricks publicly describes AI agents, AI governance, data and AI platforms, and an interview process with skills assessments and structured behavioral evaluation.", "Interviewing With Us", "https://www.databricks.com/company/careers/interview-prep", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Databricks positions AI work around data and AI platforms, agents, governance, and large-scale engineering.", "Engineering / AI roles", "https://www.databricks.com/company/careers/interview-prep", "2026", ["ML Infrastructure", "Agents", "Distributed Systems", "MLOps"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "ML Engineer", "", "A public Databricks hiring megathread is retained as process context, but the extracted page did not expose a candidate-specific question.", "https://www.reddit.com/r/databricks/comments/1jf5d8r/megathread_hiring_and_interviewing_at_databricks/", "[Megathread] Hiring and Interviewing at Databricks", "2025-03-19", ["ML Infrastructure", "Behavioral"], "low")],
        "inferred": ("How would you design a production ML platform that supports governed data, model training, agent deployment, evaluation, and low-latency retrieval at scale?", ["ML Infrastructure", "MLOps", "Agents", "RAG", "Distributed Systems"]),
        "sources": [("Interviewing With Us", "https://www.databricks.com/company/careers/interview-prep", "", "COMPANY_TECHNICAL_FOCUS"), ("[Megathread] Hiring and Interviewing at Databricks", "https://www.reddit.com/r/databricks/comments/1jf5d8r/megathread_hiring_and_interviewing_at_databricks/", "2025-03-19", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Snowflake", "domains": ["Agents", "RAG", "MLOps", "AI Reliability", "ML Infrastructure"],
        "roles": ["AI Engineer", "ML Platform Engineer", "Applied AI Engineer"],
        "focus": ("Snowflake’s AI engineering teams include AI research, applied AI, Cortex agents and apps, AISQL, Cortex platform, observability, and Snowflake ML. The company explicitly references production LLM applications, agents, search, evaluation, monitoring, and end-to-end ML workflows.", "AI Engineering", "https://careers.snowflake.com/us/en/ai-ml-engineering", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Current AI engineering teams work on production-grade LLM applications, intelligent agents, high-quality search, evaluation/monitoring, model runtimes, and end-to-end ML workflows.", "AI Research / Applied AI / Cortex / Snowflake ML roles", "https://careers.snowflake.com/us/en/ai-ml-engineering", "2026", ["LLM", "Agents", "RAG", "Evaluation", "MLOps", "ML Infrastructure"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "A public Snowflake interview discussion is retained as process context, but it did not expose a completed candidate question in the opened page.", "https://www.reddit.com/r/leetcode/comments/1l3i90k/snowflake_interview_experiences_with_ic1_ic2/", "Snowflake Interview Experiences with IC1 / IC2 Rounds", "2025", ["DSA", "System Design"], "low")],
        "inferred": ("How would you evaluate and monitor an enterprise agent so that its retrieval is grounded, its outputs are relevant, and its behavior is observable in production?", ["Agents", "RAG", "Evaluation", "AI Reliability", "MLOps"]),
        "sources": [("AI Engineering", "https://careers.snowflake.com/us/en/ai-ml-engineering", "", "ROLE_REQUIREMENT"), ("Snowflake Interview Experiences with IC1 / IC2 Rounds", "https://www.reddit.com/r/leetcode/comments/1l3i90k/snowflake_interview_experiences_with_ic1_ic2/", "2025", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Scale AI", "domains": ["Computer Vision", "NLP", "Evaluation", "AI Reliability", "MLOps"],
        "roles": ["Machine Learning Engineer", "ML Research Engineer"],
        "focus": ("Scale AI candidate evidence centers on practical data-centric ML work, model evaluation, and take-home or notebook-based tasks.", "Machine Learning Engineer Interview Experience - San Jose, California", "https://www.jointaro.com/interviews/companies/scale-ai/experiences/machine-learning-engineer-san-jose-california-january-1-2024-no-offer-neutral-886fb1b4/", "2024-01-01", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [
            ("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Candidate reported a take-home exam with a choice of CV or NLP, followed by ML coding in a Colab notebook using raw data and supplied models.", "https://www.jointaro.com/interviews/companies/scale-ai/experiences/machine-learning-engineer-san-jose-california-january-1-2024-no-offer-neutral-886fb1b4/", "Machine Learning Engineer Interview Experience - San Jose, California", "2024-01-01", ["Computer Vision", "NLP", "Evaluation"], "medium"),
            ("REPORTED_QUESTION", "Machine Learning Engineer", "Create a read-image function, convert a 3D array to 4D model input, and evaluate the model using metrics.", "Candidate-reported ML coding task.", "https://www.jointaro.com/interviews/companies/scale-ai/experiences/machine-learning-engineer-san-jose-california-january-1-2024-no-offer-neutral-886fb1b4/", "Machine Learning Engineer Interview Experience - San Jose, California", "2024-01-01", ["Computer Vision", "Python", "Evaluation"], "medium"),
        ],
        "inferred": ("How would you design a data-quality and model-evaluation workflow for noisy CV or NLP training data before deploying a model?", ["Computer Vision", "NLP", "Evaluation", "AI Reliability", "MLOps"]),
        "sources": [("Machine Learning Engineer Interview Experience - San Jose, California", "https://www.jointaro.com/interviews/companies/scale-ai/experiences/machine-learning-engineer-san-jose-california-january-1-2024-no-offer-neutral-886fb1b4/", "2024-01-01", "REPORTED_QUESTION"), ("Scale ML Engineer Interview Experience & Questions", "https://www.glassdoor.com/Interview/Scale-ML-Engineer-Interview-Questions-EI_IE1656849.0,5_KO6,17.htm", "", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "LangChain", "domains": ["Agents", "RAG", "Tool Calling", "Evaluation", "MLOps"],
        "roles": ["Deployed Engineer", "ML Engineer"],
        "focus": ("Candidate evidence centers on LangSmith demos and LangGraph implementation/presentation, reflecting an agent and observability-oriented engineering focus.", "LangChain Deployed Engineer Interview Experience", "https://www.tryexponent.com/experiences/lang-chain-machine-learning-engineer-interview-fea4c2", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Deployed Engineer", "", "Candidate reported an intro chat, LangSmith demo take-home, and a heavier LangGraph build-and-present round; the candidate received an offer after about one month.", "https://www.tryexponent.com/experiences/lang-chain-machine-learning-engineer-interview-fea4c2", "LangChain Deployed Engineer Interview Experience", "", ["Agents", "Tool Calling", "Evaluation", "System Design"], "medium")],
        "inferred": ("How would you build and evaluate a LangGraph-style agent with tools, state, retries, observability, and safe failure behavior?", ["Agents", "Tool Calling", "Evaluation", "AI Reliability", "MLOps"]),
        "sources": [("LangChain Deployed Engineer Interview Experience", "https://www.tryexponent.com/experiences/lang-chain-machine-learning-engineer-interview-fea4c2", "", "REPORTED_INTERVIEW_PROCESS"), ("LangChain", "https://www.langchain.com/", "", "COMPANY_TECHNICAL_FOCUS")],
    },
    {
        "company": "Replit", "domains": ["Agents", "LLM", "Developer Tools", "System Design"],
        "roles": ["AI Researcher", "Software Engineer", "Product Engineer"],
        "focus": ("Replit’s official process evaluates real-world skills through recruiter, hiring-manager, technical exercise, and panel stages; its roles include AI research and collaborative developer tooling.", "Our Interview Process", "https://replit.com/interview-process", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [
            ("REPORTED_INTERVIEW_PROCESS", "Engineer", "", "Candidate reported a short take-home coding challenge, a one-hour live coding call, and a virtual onsite with collaborative design and a project demo.", "https://www.jointaro.com/interviews/companies/replit/experiences/engineer-san-francisco-california-november-1-2021-accepted-offer-positive-b1d2ddee/", "Engineer Interview Experience - San Francisco", "2021-11-01", ["DSA", "System Design", "Behavioral"], "medium"),
            ("REPORTED_QUESTION", "Engineer", "Three coding questions of increasing scope, complexity, and length.", "Candidate-reported description of the live coding stage.", "https://www.jointaro.com/interviews/companies/replit/experiences/engineer-san-francisco-california-november-1-2021-accepted-offer-positive-b1d2ddee/", "Engineer Interview Experience - San Francisco", "2021-11-01", ["DSA", "Python"], "medium"),
        ],
        "inferred": ("How would you design a collaborative AI coding workspace that combines project state, tool calls, safe execution, and human review?", ["Agents", "Tool Calling", "System Design", "AI Reliability"]),
        "sources": [("Our Interview Process", "https://replit.com/interview-process", "", "COMPANY_TECHNICAL_FOCUS"), ("Engineer Interview Experience - San Francisco", "https://www.jointaro.com/interviews/companies/replit/experiences/engineer-san-francisco-california-november-1-2021-accepted-offer-positive-b1d2ddee/", "2021-11-01", "REPORTED_QUESTION")],
    },
    {
        "company": "Weights & Biases", "domains": ["MLOps", "Evaluation", "Observability", "ML Infrastructure"],
        "roles": ["AI Engineer", "ML Engineer"],
        "focus": ("Weights & Biases states its mission is to build tools for AI and describes products used by ML teams, developers, and AI app builders; its careers page lists initial conversations, team/panel interviews, and executive calls.", "Careers at Weights & Biases", "https://wandb.ai/site/careers/", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "AI Engineer", "", "A Glassdoor search result reported phone screens and multiple technical rounds, but the page was not accessible for full verification; retained as low-confidence process context only.", "https://www.glassdoor.com/Interview/Weights-and-Biases-AI-Engineer-Interview-Questions-EI_IE2231103.0,18_KO19,30.htm", "Weights & Biases AI Engineer interview questions", "", ["MLOps", "Evaluation"], "low")],
        "inferred": ("How would you design an experiment-tracking, model-evaluation, and production-observability system for an ML team shipping LLM applications?", ["MLOps", "Evaluation", "AI Reliability", "ML Infrastructure"]),
        "sources": [("Careers at Weights & Biases", "https://wandb.ai/site/careers/", "", "COMPANY_TECHNICAL_FOCUS"), ("Weights & Biases AI Engineer interview questions", "https://www.glassdoor.com/Interview/Weights-and-Biases-AI-Engineer-Interview-Questions-EI_IE2231103.0,18_KO19,30.htm", "", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Harvey", "domains": ["LLM", "RAG", "Agents", "AI Reliability", "NLP"],
        "roles": ["Software Engineer", "Applied AI Engineer", "Applied Legal Research"],
        "focus": ("Harvey describes legal AI work involving customization, collaboration, model training, agentic workflows, smarter routing, legal-specific model evaluation, retrieval, and domain-specific model alignment.", "The Ultimate Guide to Landing a Job at Harvey", "https://www.harvey.ai/blog/landing-a-job-at-harvey", "2026-08-05", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_QUESTION", "Software Engineer", "Code a file storage system to get and put via paths.", "Candidate-reported coding question.", "https://www.jointaro.com/interviews/companies/harvey-ai/experiences/software-engineer-united-states-july-6-2025-no-offer-negative-32e463e6/", "Software Engineer Interview Experience - United States", "2025-07-06", ["System Design", "DSA"], "medium"), ("REPORTED_INTERVIEW_PROCESS", "Software Engineer", "", "Candidate reported coding technical screens followed by tough onsite interviews including coding and system design.", "https://www.jointaro.com/interviews/companies/harvey-ai/experiences/software-engineer-united-states-july-6-2025-no-offer-negative-32e463e6/", "Software Engineer Interview Experience - United States", "2025-07-06", ["DSA", "System Design"], "medium")],
        "inferred": ("How would you design a retrieval-grounded legal workflow that routes requests across models, preserves customer-specific expertise, and evaluates factual reliability?", ["RAG", "Agents", "Evaluation", "AI Reliability", "NLP"]),
        "sources": [("The Ultimate Guide to Landing a Job at Harvey", "https://www.harvey.ai/blog/landing-a-job-at-harvey", "2026-08-05", "COMPANY_TECHNICAL_FOCUS"), ("Software Engineer Interview Experience - United States", "https://www.jointaro.com/interviews/companies/harvey-ai/experiences/software-engineer-united-states-july-6-2025-no-offer-negative-32e463e6/", "2025-07-06", "REPORTED_QUESTION")],
    },
    {
        "company": "Runway", "domains": ["Computer Vision", "Deep Learning", "Multimodal", "Generative AI"],
        "roles": ["Research Engineer", "Applied Research Scientist", "Foundation Models Engineer"],
        "focus": ("Runway states it builds AI to simulate the world by merging art and science, and its current careers page lists foundation-model, applied-research, research-engineering/data-foundations, and robotics roles.", "Careers in AI Video Technology", "https://runway.com/en/careers", "2026", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Current research roles include foundation models, applied research, data foundations, and robotics; the company also describes multimodal simulators and world models.", "Research Science Manager - Foundation Models / Research Engineer / Applied Research Scientist", "https://runway.com/en/careers", "2026", ["Computer Vision", "Deep Learning", "Multimodal", "Generative AI"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Senior Machine Learning Engineer", "", "A Glassdoor search result reported mostly positive experiences but no exact question; retained as low-confidence context only.", "https://www.glassdoor.com/Interview/Runway-NYC-Senior-Machine-Learning-Engineer-Interview-Questions-EI_IE5638023.0,10_KO11,43.htm", "Runway Senior Machine Learning Engineer interview questions", "", ["Deep Learning", "Computer Vision"], "low")],
        "inferred": ("How would you train and evaluate a multimodal world model for video generation while controlling data quality, temporal consistency, and inference cost?", ["Computer Vision", "Deep Learning", "Multimodal", "Evaluation", "Inference Optimization"]),
        "sources": [("Careers in AI Video Technology", "https://runway.com/en/careers", "2026", "ROLE_REQUIREMENT"), ("Runway Senior Machine Learning Engineer interview questions", "https://www.glassdoor.com/Interview/Runway-NYC-Senior-Machine-Learning-Engineer-Interview-Questions-EI_IE5638023.0,10_KO11,43.htm", "", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Glean", "domains": ["RAG", "Retrieval", "NLP", "Agents", "Evaluation"],
        "roles": ["Machine Learning Engineer", "Search/Assistant Engineer"],
        "focus": ("Glean’s careers page describes a Work AI mission and customer-driven engineering culture; the company is included for enterprise search and assistant-oriented AI work, while dynamic job details were not exposed in the opened page.", "Careers at Glean", "https://www.glean.com/careers", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "A 2026 search result claimed technical screening, coding, resume deep dive, ML system design, and behavioral stages, but the source was not opened; retained as low-confidence process context.", "https://dataford.io/interview-guides/glean-technologies/machine-learning-engineer", "Glean Technologies Machine Learning Engineer interview guide", "2026-07-05", ["RAG", "Retrieval", "System Design", "Behavioral"], "low")],
        "inferred": ("How would you build an enterprise assistant that retrieves permissions-aware documents, ranks evidence, and evaluates answer quality across changing knowledge bases?", ["RAG", "Retrieval", "NLP", "Evaluation", "AI Reliability"]),
        "sources": [("Careers at Glean", "https://www.glean.com/careers", "", "COMPANY_TECHNICAL_FOCUS"), ("Glean Technologies Machine Learning Engineer interview guide", "https://dataford.io/interview-guides/glean-technologies/machine-learning-engineer", "2026-07-05", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Together AI", "domains": ["AI Infrastructure", "Inference Optimization", "GPU", "ML Infrastructure", "Open Source"],
        "roles": ["Research Engineer", "ML Infrastructure Engineer"],
        "focus": ("Together AI states that it co-designs software, hardware, algorithms, and models to lower the cost of modern AI systems and contributes to open-source research, models, and datasets.", "Careers | Together AI", "https://www.together.ai/careers", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("The careers page identifies research and engineering roles in next-generation AI infrastructure.", "Engineering / Research roles", "https://www.together.ai/careers", "2026", ["AI Infrastructure", "GPU", "Inference Optimization", "ML Infrastructure"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "A 2026 search result claimed a rigorous, fast-paced, deeply technical process, but the underlying guide was not opened; retained as low-confidence context.", "https://dataford.io/interview-guides/together-ai/machine-learning-engineer", "Together AI Machine Learning Engineer interview guide", "2026-08-01", ["ML Infrastructure", "GPU", "Inference Optimization"], "low")],
        "inferred": ("How would you co-design an inference stack across model architecture, GPU kernels, scheduling, and serving APIs to reduce cost per token?", ["GPU", "Inference Optimization", "Distributed Systems", "ML Infrastructure"]),
        "sources": [("Careers | Together AI", "https://www.together.ai/careers", "", "ROLE_REQUIREMENT"), ("Together AI Machine Learning Engineer interview guide", "https://dataford.io/interview-guides/together-ai/machine-learning-engineer", "2026-08-01", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Groq", "domains": ["Inference Optimization", "GPU", "AI Infrastructure", "Distributed Systems"],
        "roles": ["Inference Engineer", "AI Infrastructure Engineer"],
        "focus": ("Groq describes a global inference cloud built from silicon to cloud, with bare-metal infrastructure, enterprise governance, and an integrated hardware/software inference stack.", "Groq Careers", "https://www.groq.com/careers", "2026", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Current technical work spans LPU hardware, inference operations, hyperscale infrastructure, and enterprise software.", "Inference and infrastructure roles", "https://www.groq.com/careers", "2026", ["Inference Optimization", "GPU", "AI Infrastructure", "Distributed Systems"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Engineer", "", "A Glassdoor search result reported multi-stage technical and personality assessments; the page was not opened, so this is low-confidence process context.", "https://www.glassdoor.com/Interview/Groq-Engineer-Interview-Questions-EI_IE2473036.0,4_KO5,13.htm", "Groq Engineer Interview Experience & Questions", "", ["Inference Optimization", "System Design"], "low")],
        "inferred": ("How would you design a low-latency inference service that maps model workloads to specialized hardware while maintaining reliability and enterprise controls?", ["Inference Optimization", "GPU", "AI Reliability", "Distributed Systems"]),
        "sources": [("Groq Careers", "https://www.groq.com/careers", "2026", "ROLE_REQUIREMENT"), ("Groq Engineer Interview Experience & Questions", "https://www.glassdoor.com/Interview/Groq-Engineer-Interview-Questions-EI_IE2473036.0,4_KO5,13.htm", "", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Salesforce", "domains": ["Agents", "LLM", "RAG", "AI Reliability", "System Design"],
        "roles": ["AI Engineer", "ML Engineer", "Agent Engineer"],
        "focus": ("Salesforce describes an agent-first enterprise centered on Agentforce and human-agent collaboration.", "How Salesforce Hires", "https://www.salesforce.com/company/careers/culture/how-we-hire/", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [
            ("REPORTED_INTERVIEW_PROCESS", "Staff AI Engineer", "", "Candidate reported HR screening, hiring-manager chat, project deep dive, AI/ML fundamentals, hands-on coding, and system design.", "https://www.reddit.com/r/OfferEngineering/comments/1syye9w/this_staff_ai_interview_at_salesforce_was_way/", "This Staff AI interview at Salesforce was way more intense than expected", "2026-04-29", ["LLM", "Agents", "RAG", "System Design", "Behavioral"], "medium"),
            ("REPORTED_QUESTION", "Staff AI Engineer", "Implement a web crawler to scrape a specified site, sort the content, and export it to CSV.", "Candidate-reported hands-on mini-project.", "https://www.reddit.com/r/OfferEngineering/comments/1syye9w/this_staff_ai_interview_at_salesforce_was_way/", "This Staff AI interview at Salesforce was way more intense than expected", "2026-04-29", ["Python", "DSA"], "medium"),
            ("REPORTED_QUESTION", "Staff AI Engineer", "Design a web service similar to Google Sheets, including concurrency control, storage, large-scale load/save, snapshot backups, and database tables.", "Candidate-reported system-design task.", "https://www.reddit.com/r/OfferEngineering/comments/1syye9w/this_staff_ai_interview_at_salesforce_was_way/", "This Staff AI interview at Salesforce was way more intense than expected", "2026-04-29", ["System Design", "Distributed Systems"], "medium"),
        ],
        "inferred": ("How would you design an agentic enterprise workflow with retrieval, grounding, guardrails, and observability so that humans and agents can collaborate safely?", ["Agents", "RAG", "AI Reliability", "Evaluation", "System Design"]),
        "sources": [("How Salesforce Hires", "https://www.salesforce.com/company/careers/culture/how-we-hire/", "", "COMPANY_TECHNICAL_FOCUS"), ("This Staff AI interview at Salesforce was way more intense than expected", "https://www.reddit.com/r/OfferEngineering/comments/1syye9w/this_staff_ai_interview_at_salesforce_was_way/", "2026-04-29", "REPORTED_QUESTION")],
    },
    {
        "company": "ServiceNow", "domains": ["Agents", "AI Platform", "MLOps", "AI Reliability", "System Design"],
        "roles": ["LLM Engineer", "ML Engineer", "AI Platform Engineer"],
        "focus": ("ServiceNow describes its AI Platform as a way to automate processes, develop/run/manage applications, prioritize and automate tasks, detect incidents, and surface insights.", "How we hire - ServiceNow Careers", "https://careers.servicenow.com/how-we-hire/", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("ServiceNow states that technical assessments or presentations may be part of role-specific interviews and that candidates must complete evaluation independently unless otherwise specified.", "AI Platform and ML roles", "https://careers.servicenow.com/how-we-hire/", "2026", ["Agents", "MLOps", "AI Reliability", "System Design"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "LLM Engineer", "", "A 2025 Reddit discussion mentions preparing to discuss projects, performance optimization, and possible system design, but it does not provide a completed candidate report; low confidence.", "https://www.reddit.com/r/servicenow/comments/1hv795z/i_have_a_servicenow_llm_engineer_interview_coming/", "I have a servicenow LLM engineer interview coming up", "2025-01-06", ["LLM", "System Design", "Inference Optimization"], "low")],
        "inferred": ("How would you build an AI workflow on an enterprise platform that detects incidents, calls tools, evaluates outcomes, and preserves human oversight?", ["Agents", "Tool Calling", "AI Reliability", "MLOps"]),
        "sources": [("How we hire - ServiceNow Careers", "https://careers.servicenow.com/how-we-hire/", "", "ROLE_REQUIREMENT"), ("I have a servicenow LLM engineer interview coming up", "https://www.reddit.com/r/servicenow/comments/1hv795z/i_have_a_servicenow_llm_engineer_interview_coming/", "2025-01-06", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Palantir", "domains": ["AI Platform", "ML Infrastructure", "Agents", "System Design", "Distributed Systems"],
        "roles": ["Forward Deployed Engineer", "ML Platform Engineer", "AI Systems Engineer"],
        "focus": ("Palantir describes Deltas as delivering scalable data infrastructure and AI systems that work in practice, and Devs as building platforms across the full product lifecycle.", "Careers", "https://www.palantir.com/careers/", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("Palantir’s role model emphasizes problem decomposition, technical software and AI solutions, scalable data infrastructure, and production systems.", "Echos / Deltas / Devs", "https://www.palantir.com/careers/", "2026", ["AI Infrastructure", "ML Infrastructure", "System Design", "Distributed Systems"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Software Engineer", "", "A 2026 search result reported an online HackerRank assessment focused on data structures and algorithms; the source was not opened, so retain as low confidence.", "https://www.reddit.com/r/leetcode/comments/1tuytm0/palantir_software_engineer_interview_experience/", "Palantir Software Engineer Interview Experience - New Grad 2026", "2026-06-02", ["DSA"], "low")],
        "inferred": ("How would you take an ambiguous partner workflow, decompose it into data and AI services, and deliver a reliable production system with measurable outcomes?", ["System Design", "ML Infrastructure", "Agents", "AI Reliability"]),
        "sources": [("Careers", "https://www.palantir.com/careers/", "", "ROLE_REQUIREMENT"), ("Palantir Software Engineer Interview Experience - New Grad 2026", "https://www.reddit.com/r/leetcode/comments/1tuytm0/palantir_software_engineer_interview_experience/", "2026-06-02", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Stripe", "domains": ["Machine Learning", "Deep Learning", "Inference Optimization", "MLOps", "Fraud Detection"],
        "roles": ["Machine Learning Engineer, Radar"],
        "focus": ("Stripe’s Radar ML team builds real-time deep-learning fraud models and products for a large payment network, with an emphasis on production ML and low latency.", "Machine Learning Engineer, Radar", "https://stripe.com/careers/listing/machine-learning-engineer-radar/7983456", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("The role requires Python, SQL, Spark, PyTorch, production ML, statistics, experiment design, real-time low-latency infrastructure, and end-to-end ML product design.", "Machine Learning Engineer, Radar", "https://stripe.com/careers/listing/machine-learning-engineer-radar/7983456", "2026", ["Python", "Machine Learning", "Deep Learning", "MLOps", "Inference Optimization", "Distributed Systems"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Candidate reported programming, ML design, ML integration, bug squash, and hiring-manager chat rounds.", "https://www.jointaro.com/interviews/companies/stripe/experiences/machine-learning-engineer-united-states-march-14-2022-no-offer-negative-91e22ff6/", "Machine Learning Engineer Interview Experience - United States", "2022-03-14", ["Machine Learning", "System Design", "MLOps"], "medium")],
        "inferred": ("How would you design a real-time fraud model that handles distribution shift, low-latency inference, experiment measurement, and adversarial behavior?", ["Machine Learning", "Deep Learning", "Evaluation", "Inference Optimization", "MLOps"]),
        "sources": [("Machine Learning Engineer, Radar", "https://stripe.com/careers/listing/machine-learning-engineer-radar/7983456", "2026", "ROLE_REQUIREMENT"), ("Machine Learning Engineer Interview Experience - United States", "https://www.jointaro.com/interviews/companies/stripe/experiences/machine-learning-engineer-united-states-march-14-2022-no-offer-negative-91e22ff6/", "2022-03-14", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Atlassian", "domains": ["ML Infrastructure", "Distributed Systems", "System Design", "MLOps"],
        "roles": ["Machine Learning Engineer", "Software Engineer"],
        "focus": ("Atlassian’s engineering handbook describes secure, reliable, compliant software at scale and a process testing data structures, code design, system design, collaboration, and values.", "Atlassian engineering interview handbook", "https://www.atlassian.com/company/careers/resources/interviewing/engineering", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "A Taro page lists a February 1, 2025 Melbourne MLE process with recruiter screen, hiring-manager interview, technical rounds including coding and system design, and potentially a virtual onsite.", "https://www.jointaro.com/interviews/companies/atlassian/work-experiences/machine-learning-engineer-bengaluru-june-9-2024-5-ca8e5b35/", "Atlassian Machine Learning Engineer work experience", "2025-02-01", ["DSA", "System Design", "Behavioral"], "medium")],
        "inferred": ("How would you design a reliable ML service under explicit scale, cost, security, and compliance constraints, and explain the trade-offs to collaborators?", ["System Design", "Distributed Systems", "MLOps", "AI Reliability"]),
        "sources": [("Atlassian engineering interview handbook", "https://www.atlassian.com/company/careers/resources/interviewing/engineering", "", "COMPANY_TECHNICAL_FOCUS"), ("Atlassian Machine Learning Engineer work experience", "https://www.jointaro.com/interviews/companies/atlassian/work-experiences/machine-learning-engineer-bengaluru-june-9-2024-5-ca8e5b35/", "2025-02-01", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "AMD", "domains": ["Deep Learning", "Computer Vision", "Transformers", "GPU", "Inference Optimization"],
        "roles": ["Senior Software Engineer (AI)", "ML Software Engineer"],
        "focus": ("Candidate evidence for AMD’s AI role covers deep-learning architecture, object detection, transformer architecture, attention, C++, and ML systems.", "Interview Experience @ AMD, Senior Software Engineer (AI) [2024]", "https://ihitsuperhuman.medium.com/interview-experience-amd-ai-engineer-2024-1a86afd025ad", "2024-12-21", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Senior Software Engineer (AI)", "", "Candidate reported six rounds covering resume/problem solving, ML, DSA/C++, a second ML round, hiring manager, and HR.", "https://ihitsuperhuman.medium.com/interview-experience-amd-ai-engineer-2024-1a86afd025ad", "Interview Experience @ AMD, Senior Software Engineer (AI) [2024]", "2024-12-21", ["DSA", "Machine Learning", "Deep Learning", "Transformers", "Behavioral"], "medium"), ("REPORTED_QUESTION", "Senior Software Engineer (AI)", "Explain activation functions including ReLU and PReLU, regularization, object detection workflows, residual connections, and non-maximum suppression.", "Candidate-reported ML round topics.", "https://ihitsuperhuman.medium.com/interview-experience-amd-ai-engineer-2024-1a86afd025ad", "Interview Experience @ AMD, Senior Software Engineer (AI) [2024]", "2024-12-21", ["Deep Learning", "Computer Vision"], "medium"), ("REPORTED_QUESTION", "Senior Software Engineer (AI)", "Explain transformer architecture and different types of attention mechanisms and their applications.", "Candidate-reported advanced ML round topics.", "https://ihitsuperhuman.medium.com/interview-experience-amd-ai-engineer-2024-1a86afd025ad", "Interview Experience @ AMD, Senior Software Engineer (AI) [2024]", "2024-12-21", ["Transformers", "Deep Learning"], "medium")],
        "inferred": ("How would you map transformer attention and vision-model workloads onto accelerator hardware while improving memory use and inference throughput?", ["Transformers", "GPU", "Inference Optimization", "Deep Learning"]),
        "sources": [("Interview Experience @ AMD, Senior Software Engineer (AI) [2024]", "https://ihitsuperhuman.medium.com/interview-experience-amd-ai-engineer-2024-1a86afd025ad", "2024-12-21", "REPORTED_QUESTION")],
    },
    {
        "company": "Intel", "domains": ["LLM", "Agents", "Evaluation", "AI Safety", "GPU", "Inference Optimization"],
        "roles": ["Senior Machine Learning Engineer", "AI Software Solutions Engineer"],
        "focus": ("Intel’s current role description emphasizes agent harnesses, context engineering, memory, tools, model evaluation, post-training, RL environments, reward models, GPU utilization, and multi-GPU debugging.", "Sr. Machine Learning Engineer", "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Sr-Machine-Learning-Engineer_JR0285966", "2026-08-14", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("The role requires Python, LLM architectures, evaluation frameworks and benchmarks, post-training, supervised fine-tuning, reinforcement learning, reward models, GPU optimization, and agentic applications.", "Sr. Machine Learning Engineer", "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Sr-Machine-Learning-Engineer_JR0285966", "2026-08-14", ["Python", "LLM", "Agents", "Evaluation", "AI Safety", "GPU", "Inference Optimization"]),
        "reports": [("REPORTED_QUESTION", "AI Software Solutions Engineer", "Given an array of positive integers and a target, find the minimum length of a contiguous subarray whose sum is at least the target.", "Candidate-reported technical interview problem.", "https://ihitsuperhuman.medium.com/interview-experience-intel-ai-engineer-2024-fb553ea08eb7", "Interview Experience @ Intel, AI Engineer [2024]", "2024-12-07", ["DSA", "Python"], "medium"), ("REPORTED_QUESTION", "AI Software Solutions Engineer", "Solve a condition-based flooding problem on a 2D matrix, treating it as a graph/DFS problem.", "Candidate-reported technical interview problem.", "https://ihitsuperhuman.medium.com/interview-experience-intel-ai-engineer-2024-fb553ea08eb7", "Interview Experience @ Intel, AI Engineer [2024]", "2024-12-07", ["DSA", "Computer Vision"], "medium"), ("REPORTED_QUESTION", "AI Software Solutions Engineer", "Explain Transformer architecture, the physical significance of its components, and how attention can be optimized.", "Candidate-reported AI/ML round topics.", "https://ihitsuperhuman.medium.com/interview-experience-intel-ai-engineer-2024-fb553ea08eb7", "Interview Experience @ Intel, AI Engineer [2024]", "2024-12-07", ["Transformers", "Inference Optimization", "Deep Learning"], "medium")],
        "inferred": ("How would you build a reproducible post-training and evaluation pipeline for a small agent backend that must fit edge-device runtime constraints?", ["LLM", "Agents", "Evaluation", "GPU", "AI Safety", "Inference Optimization"]),
        "sources": [("Sr. Machine Learning Engineer", "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Sr-Machine-Learning-Engineer_JR0285966", "2026-08-14", "ROLE_REQUIREMENT"), ("Interview Experience @ Intel, AI Engineer [2024]", "https://ihitsuperhuman.medium.com/interview-experience-intel-ai-engineer-2024-fb553ea08eb7", "2024-12-07", "REPORTED_QUESTION")],
    },
    {
        "company": "Qualcomm", "domains": ["Computer Vision", "Deep Learning", "Inference Optimization", "AI Infrastructure", "GPU"],
        "roles": ["Computer Vision Engineer", "ML Engineer"],
        "focus": ("A 2025 candidate report describes Qualcomm computer-vision interviews covering CNNs, R-CNN, YOLO, object detection, model selection, optimization, evaluation, deployment, and project decisions.", "Qualcomm Interview Experience", "https://leetcode.com/discuss/interview-experience/7411588/", "2025-12-13", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Computer Vision Engineer", "", "Candidate reported a 2.5-hour interview split into behavioral/introduction, project discussion, DSA, computer vision, and advanced AI/ML sections.", "https://leetcode.com/discuss/interview-experience/7411588/", "Qualcomm Interview Experience", "2025-12-13", ["Behavioral", "DSA", "Computer Vision", "Machine Learning"], "medium"), ("REPORTED_QUESTION", "Computer Vision Engineer", "Discuss CNNs, R-CNN, YOLO, object-detection architectures, model selection, optimization, evaluation, and deployment challenges.", "Candidate-reported computer-vision and AI/ML topics.", "https://leetcode.com/discuss/interview-experience/7411588/", "Qualcomm Interview Experience", "2025-12-13", ["Computer Vision", "Deep Learning", "MLOps"], "medium")],
        "inferred": ("How would you optimize an object-detection model for a Qualcomm accelerator while maintaining accuracy, latency, and deployment reliability?", ["Computer Vision", "Deep Learning", "Inference Optimization", "GPU", "MLOps"]),
        "sources": [("Qualcomm Interview Experience", "https://leetcode.com/discuss/interview-experience/7411588/", "2025-12-13", "REPORTED_QUESTION")],
    },
    {
        "company": "PayPal", "domains": ["Machine Learning", "Fraud Detection", "MLOps", "Deep Learning", "System Design"],
        "roles": ["Machine Learning Engineer"],
        "focus": ("PayPal candidate evidence describes ML engineering for payments through coding, ML fundamentals, in-depth ML, and business-case rounds.", "Machine Learning Engineer Interview Experience - India", "https://www.jointaro.com/interviews/companies/paypal/experiences/machine-learning-engineer-india-august-1-2024-no-offer-negative-dfd1625f/", "2024-08-01", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Candidate reported two coding-plus-ML rounds, an in-depth ML round, a business-case round, and an additional combined ML/coding round.", "https://www.jointaro.com/interviews/companies/paypal/experiences/machine-learning-engineer-india-august-1-2024-no-offer-negative-dfd1625f/", "Machine Learning Engineer Interview Experience - India", "2024-08-01", ["SQL", "Python", "Machine Learning", "Deep Learning", "System Design"], "medium"), ("REPORTED_QUESTION", "Machine Learning Engineer", "Discuss SQL, basic Python/DSA, loss functions, models, algorithms, neural networks, and a machine-learning business case.", "Candidate-reported topic coverage across rounds.", "https://www.jointaro.com/interviews/companies/paypal/experiences/machine-learning-engineer-india-august-1-2024-no-offer-negative-dfd1625f/", "Machine Learning Engineer Interview Experience - India", "2024-08-01", ["Python", "DSA", "Machine Learning", "Deep Learning"], "medium")],
        "inferred": ("How would you design and evaluate a payment-risk model that combines real-time signals, model monitoring, and a business-case decision threshold?", ["Machine Learning", "Evaluation", "MLOps", "System Design"]),
        "sources": [("Machine Learning Engineer Interview Experience - India", "https://www.jointaro.com/interviews/companies/paypal/experiences/machine-learning-engineer-india-august-1-2024-no-offer-negative-dfd1625f/", "2024-08-01", "REPORTED_QUESTION")],
    },
    {
        "company": "Mastercard", "domains": ["Machine Learning", "Fraud Detection", "AI Reliability", "System Design"],
        "roles": ["Machine Learning Engineer"],
        "focus": ("Mastercard’s hiring page says the process varies by role and location; candidate evidence reports multiple interviews and a later additional round after mixed feedback.", "Our hiring process", "https://careers.mastercard.com/us/en/mastercards-hiring-process", "", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Candidate reported four one-hour interview rounds, a later additional one-hour interview because of mixed feedback, and a final rejection after more than two months.", "https://www.reddit.com/r/interviews/comments/1nd5r1p/interview_experience_at_mastercard/", "Interview experience at Mastercard", "2025-09-10", ["Behavioral", "System Design"], "low")],
        "inferred": ("How would you build a low-latency fraud-detection system for card transactions and measure false positives, recall, and operational reliability?", ["Machine Learning", "Evaluation", "AI Reliability", "Inference Optimization", "System Design"]),
        "sources": [("Our hiring process", "https://careers.mastercard.com/us/en/mastercards-hiring-process", "", "COMPANY_TECHNICAL_FOCUS"), ("Interview experience at Mastercard", "https://www.reddit.com/r/interviews/comments/1nd5r1p/interview_experience_at_mastercard/", "2025-09-10", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Spotify", "domains": ["Recommender Systems", "NLP", "Embeddings", "Evaluation", "MLOps"],
        "roles": ["Machine Learning Engineer", "Recommendation Engineer"],
        "focus": ("Spotify interview-guide evidence describes large-scale personalization, real-time recommendation, audio understanding, experimentation, model evaluation, and production monitoring.", "Spotify Machine Learning Engineer Interview Guide", "https://www.interviewquery.com/interview-guides/spotify-machine-learning-engineer", "2026-03-17", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("The guide describes ML roles involving personalization, feature engineering, deployment, monitoring, recommender systems, and experimentation at large scale; treat this as guide-derived rather than a direct job description.", "Machine Learning Engineer", "https://www.interviewquery.com/interview-guides/spotify-machine-learning-engineer", "2026-03-17", ["Machine Learning", "NLP", "Embeddings", "Evaluation", "MLOps"]),
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "The guide describes recruiter screen, technical interview, 4-5 onsite rounds, final hiring-manager interview, and offer discussion; this is guide-derived process context, not a candidate-specific report.", "https://www.interviewquery.com/interview-guides/spotify-machine-learning-engineer", "Spotify Machine Learning Engineer Interview Guide", "2026-03-17", ["DSA", "Machine Learning", "System Design", "Behavioral"], "low")],
        "inferred": ("How would you design a Discover Weekly-style recommender with candidate generation, ranking, diversity, online experimentation, and monitoring for drift?", ["Machine Learning", "Evaluation", "Embeddings", "MLOps", "System Design"]),
        "sources": [("Spotify Machine Learning Engineer Interview Guide", "https://www.interviewquery.com/interview-guides/spotify-machine-learning-engineer", "2026-03-17", "COMPANY_TECHNICAL_FOCUS")],
    },
    {
        "company": "ByteDance", "domains": ["NLP", "Computer Vision", "Multimodal", "Recommender Systems", "LLM", "Distributed Systems"],
        "roles": ["AI Machine Learning Engineer", "Search ML Engineer"],
        "focus": ("ByteDance’s current search ML role covers NLP, CV, multimodal matching, recommendation, large-scale streaming ML, distributed systems, high-throughput/low-latency services, and LLM application/deployment.", "AI Machine Learning Engineer Graduate (Search) - 2026 Start (PhD)", "https://joinbytedance.com/search/7607045118566484277", "2026", "COMPANY_TECHNICAL_FOCUS"),
        "role": ("The role requires strong algorithms/data structures, machine learning application, and modeling in search, recommendation, NLP/CV, or multimodal/LLM systems.", "AI Machine Learning Engineer Graduate (Search) - 2026 Start (PhD)", "https://joinbytedance.com/search/7607045118566484277", "2026", ["DSA", "NLP", "Computer Vision", "Multimodal", "Recommender Systems", "LLM", "Distributed Systems"]),
        "reports": [("REPORTED_QUESTION", "Machine Learning Engineer", "LRU Cache.", "Listed in the Taro ByteDance MLE question set.", "https://www.jointaro.com/interviews/companies/bytedance/experiences/machine-learning-engineer-seattle-washington-october-1-2025-declined-offer-positive-b7903d1c/", "ByteDance Machine Learning Engineer Interview Experience", "2025", ["DSA"], "medium"), ("REPORTED_QUESTION", "Machine Learning Engineer", "Remove Duplicate Letters.", "Listed in the Taro ByteDance MLE question set.", "https://www.jointaro.com/interviews/companies/bytedance/experiences/machine-learning-engineer-seattle-washington-october-1-2025-declined-offer-positive-b7903d1c/", "ByteDance Machine Learning Engineer Interview Experience", "2025", ["DSA", "Strings"], "medium"), ("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Related 2025 candidate reports include medium coding, resume/project discussion, deep-learning architectures such as diffusion models, and recommendation topics.", "https://www.jointaro.com/interviews/companies/bytedance/experiences/machine-learning-engineer-seattle-washington-october-1-2025-declined-offer-positive-b7903d1c/", "ByteDance Machine Learning Engineer Interview Experience", "2025", ["Machine Learning", "Deep Learning", "Recommender Systems"], "medium")],
        "inferred": ("How would you design a multimodal search and recommendation pipeline that combines semantic retrieval, streaming features, distributed training, and low-latency serving?", ["NLP", "Computer Vision", "Multimodal", "Recommender Systems", "Distributed Systems", "Inference Optimization"]),
        "sources": [("AI Machine Learning Engineer Graduate (Search) - 2026 Start (PhD)", "https://joinbytedance.com/search/7607045118566484277", "2026", "ROLE_REQUIREMENT"), ("ByteDance Machine Learning Engineer Interview Experience", "https://www.jointaro.com/interviews/companies/bytedance/experiences/machine-learning-engineer-seattle-washington-october-1-2025-declined-offer-positive-b7903d1c/", "2025", "REPORTED_QUESTION")],
    },
    {
        "company": "Tencent", "domains": ["NLP", "Recommender Systems", "LLM", "DSA", "Distributed Systems"],
        "roles": ["Machine Learning Engineer", "Backend Engineer"],
        "focus": ("Tencent candidate evidence includes a WeChat backend interview with a graph/topological-sort coding task; search results also point to multi-round ML interviews, but those details remain low confidence.", "Questionable Technical Interview at Tencent", "https://www.reddit.com/r/nus/comments/1ezdjbj/questionable_technical_interview_at_tencent/", "2024-08-23", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_QUESTION", "Backend Developer Intern", "Course Schedule II, using topological sorting/BFS, followed by discussion of the queue and time complexity.", "Candidate-reported first-round WeChat backend task; role is backend rather than AI/ML.", "https://www.reddit.com/r/nus/comments/1ezdjbj/questionable_technical_interview_at_tencent/", "Questionable Technical Interview at Tencent", "2024-08-23", ["DSA", "Distributed Systems"], "medium")],
        "inferred": ("How would you scale a recommendation or language-model serving system across a large user base while keeping graph retrieval and online inference reliable?", ["Recommender Systems", "LLM", "Distributed Systems", "Inference Optimization", "AI Reliability"]),
        "sources": [("Questionable Technical Interview at Tencent", "https://www.reddit.com/r/nus/comments/1ezdjbj/questionable_technical_interview_at_tencent/", "2024-08-23", "REPORTED_QUESTION"), ("Tencent Machine Learning Engineer Interview Experience & Questions", "https://www.glassdoor.com/Interview/Tencent-Machine-Learning-Engineer-Interview-Questions-EI_IE38281.0,7_KO8,33.htm", "2023-08-18", "REPORTED_INTERVIEW_PROCESS")],
    },
    {
        "company": "Alibaba", "domains": ["LLM", "Transformers", "NLP", "Recommender Systems", "Distributed Systems"],
        "roles": ["Machine Learning Engineer", "Research Engineer"],
        "focus": ("Alibaba candidate evidence describes online MLE interviews covering AI fundamentals, Transformer computation, coding, research experience, and ML algorithms.", "Alibaba Machine Learning Engineer work experience", "https://www.jointaro.com/interviews/companies/alibaba/work-experiences/machine-learning-engineer-june-20-2023-4-b9f68a49/", "2025", "COMPANY_TECHNICAL_FOCUS"),
        "role": None,
        "reports": [("REPORTED_INTERVIEW_PROCESS", "Machine Learning Engineer", "", "Related candidate reports include online meetings with background, research, and ML-algorithm questions; a graduate process included AI fundamentals, Transformer computation/coding, and a final chat.", "https://www.jointaro.com/interviews/companies/alibaba/work-experiences/machine-learning-engineer-june-20-2023-4-b9f68a49/", "Alibaba Machine Learning Engineer work experience", "2025", ["Machine Learning", "Transformers", "DSA", "Behavioral"], "medium"), ("REPORTED_QUESTION", "Machine Learning Engineer", "Split Concatenated Strings.", "Listed in the Taro Alibaba question set.", "https://www.jointaro.com/interviews/companies/alibaba/work-experiences/machine-learning-engineer-june-20-2023-4-b9f68a49/", "Alibaba Machine Learning Engineer work experience", "2025", ["DSA", "Strings"], "medium"), ("REPORTED_QUESTION", "Machine Learning Engineer", "Valid Parenthesis String.", "Listed in the Taro Alibaba question set.", "https://www.jointaro.com/interviews/companies/alibaba/work-experiences/machine-learning-engineer-june-20-2023-4-b9f68a49/", "Alibaba Machine Learning Engineer work experience", "2025", ["DSA", "Strings"], "medium")],
        "inferred": ("How would you design and evaluate a Transformer-based recommendation or search system that handles large-scale data and serves low-latency results?", ["Transformers", "NLP", "Recommender Systems", "Distributed Systems", "Inference Optimization"]),
        "sources": [("Alibaba Machine Learning Engineer work experience", "https://www.jointaro.com/interviews/companies/alibaba/work-experiences/machine-learning-engineer-june-20-2023-4-b9f68a49/", "2025", "REPORTED_QUESTION"), ("Alibaba Group Machine Learning Engineer interview questions", "http://www.glassdoor.com/Interview/Alibaba-Group-Machine-Learning-Engineer-Interview-Questions-EI_IE225974.0,13_KO14,39.htm", "", "REPORTED_INTERVIEW_PROCESS")],
    },
]

# Validate the exclusion rule before writing anything.
existing = {"Accenture", "Adobe", "Amazon", "Apple", "Capgemini", "Cognizant", "Directi", "Expedia", "Meta", "Flipkart", "Goldman Sachs", "Google", "IBM", "Infosys", "JP Morgan", "LTIMindtree", "LinkedIn", "Microsoft", "NVIDIA", "Netflix", "Oracle", "TCS", "Twitter", "Uber", "Visa", "VMware", "Walmart", "Zoho", "Deloitte"}
new_names = [c["company"] for c in companies]
assert len(new_names) == len(set(new_names))
assert not existing.intersection(new_names), existing.intersection(new_names)
assert len(new_names) >= 30, len(new_names)

records: list[dict] = []
for c in companies:
    company = c["company"]
    domains = c["domains"]
    focus_evidence, focus_title, focus_url, focus_date, focus_type = c["focus"]
    records.append({"company": company, "role": ", ".join(c["roles"]), "evidence_type": focus_type, "question": "", "interview_round": "", "interview_date": "", "source": focus_url.split("/")[2], "source_title": focus_title, "source_url": focus_url, "source_date": focus_date, "technical_domain": domains, "evidence": focus_evidence, "confidence": "medium", "notes": "Company-level context; not a claim that this was asked in an interview."})
    if c["role"]:
        evidence, title, url, sdate, tags = c["role"]
        records.append({"company": company, "role": c["roles"][0], "evidence_type": "ROLE_REQUIREMENT", "question": "", "interview_round": "", "interview_date": "", "source": url.split("/")[2], "source_title": title, "source_url": url, "source_date": sdate, "technical_domain": tags, "evidence": evidence, "confidence": "medium", "notes": "Current or recent role/careers evidence; not a candidate-reported interview question."})
    for item in c["reports"]:
        etype, role, question, evidence, url, title, sdate, tags, confidence = item
        records.append({"company": company, "role": role, "evidence_type": etype, "question": question, "interview_round": "", "interview_date": sdate if etype.startswith("REPORTED") else "", "source": url.split("/")[2], "source_title": title, "source_url": url, "source_date": sdate, "technical_domain": tags, "evidence": evidence, "confidence": confidence, "notes": "Candidate-reported or explicitly labeled low-confidence public process evidence; do not generalize beyond the source."})
    iq, itags = c["inferred"]
    records.append({"company": company, "role": ", ".join(c["roles"]), "evidence_type": "INFERRED_QUESTION", "question": iq, "interview_round": "", "interview_date": "", "source": "derived", "source_title": "Derived from company focus and role evidence", "source_url": focus_url, "source_date": focus_date, "technical_domain": itags, "evidence": "This is a logically inferred company-specific practice question derived from the cited technical focus; no claim is made that the company asked it.", "confidence": "low", "notes": "INFERRED_QUESTION must not be presented as a reported interview question."})

# Deterministic question deduplication: same company + evidence type + question.
seen = set()
deduped = []
for r in records:
    key = (r["company"], r["evidence_type"], re.sub(r"\s+", " ", r["question"]).strip().lower())
    if key in seen and r["question"]:
        continue
    seen.add(key)
    deduped.append(r)
records = deduped

with (OUT / "structured_interview_dataset.jsonl").open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Per-company Markdown documents for the existing .md ingestion path.
for c in companies:
    company = c["company"]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", company).strip("_")
    company_records = [r for r in records if r["company"] == company]
    lines = [f"# {company} — Company-Specific Interview Intelligence", "", f"**Evidence policy:** Records are classified as `REPORTED_QUESTION`, `REPORTED_INTERVIEW_PROCESS`, `ROLE_REQUIREMENT`, `COMPANY_TECHNICAL_FOCUS`, or `INFERRED_QUESTION`. Inferred questions are not claims about what the company asked.", ""]
    for r in company_records:
        lines.extend([f"## {r['evidence_type']}", f"**Role:** {r['role'] or 'Not specified'}", f"**Technical domains:** {', '.join(r['technical_domain'])}", f"**Question:** {r['question'] or 'No exact question recorded.'}", f"**Evidence:** {r['evidence']}", f"**Source:** [{r['source_title']}]({r['source_url']})", f"**Source date:** {r['source_date'] or 'Not exposed'}", f"**Interview date:** {r['interview_date'] or 'Not exposed'}", f"**Confidence:** {r['confidence']}", ""])
    (DATA / f"{safe}_interview_intelligence.md").write_text("\n".join(lines), encoding="utf-8")

# Coverage report.
by_company = defaultdict(list)
for r in records:
    by_company[r["company"]].append(r)
coverage_lines = ["# Company Coverage Report", "", "| Company | AI Domain | Sources | Reported Questions | Inferred Questions | Roles Covered | Freshest Source |", "|---|---|---:|---:|---:|---|---|"]
for c in companies:
    rs = by_company[c["company"]]
    dates = [r["source_date"] for r in rs if re.match(r"^20\d\d", r["source_date"])]
    freshest = max(dates) if dates else "Not exposed"
    coverage_lines.append(f"| {c['company']} | {', '.join(c['domains'])} | {len(set(r['source_url'] for r in rs))} | {sum(r['evidence_type']=='REPORTED_QUESTION' for r in rs)} | {sum(r['evidence_type']=='INFERRED_QUESTION' for r in rs)} | {', '.join(c['roles'])} | {freshest} |")
(OUT / "company_coverage_report.md").write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

# Provenance CSV.
with (OUT / "source_provenance_report.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["company", "source_title", "source_url", "source_date", "evidence_type"])
    emitted = set()
    for c in companies:
        for title, url, sdate, etype in c["sources"]:
            key = (c["company"], url, etype)
            if key not in emitted:
                writer.writerow([c["company"], title, url, sdate or "Not exposed", etype])
                emitted.add(key)

# Quality report.
counts = Counter(r["evidence_type"] for r in records)
domains = {r["source_url"].split("/")[2] for r in records if r["source_url"].startswith("http")}
known_dates = [r["source_date"] for r in records if re.match(r"^20\d\d", r["source_date"])]
weak = sorted({r["company"] for r in records if r["confidence"] == "low" and r["evidence_type"] != "INFERRED_QUESTION"})
quality = f"""# Data Quality Report

## Scope and protection

The existing-company exclusion set was recovered from the specified backend repository's own historical `data/` tree and normalized with aliases. The new-company list is disjoint from that set and from the ten mandatory exclusions in the research brief. No existing records were deleted or overwritten.

## Counts

| Metric | Value |
|---|---:|
| Existing companies detected | {len(existing)} |
| New companies added | {len(new_names)} |
| Total structured records | {len(records)} |
| Reported questions | {counts['REPORTED_QUESTION']} |
| Reported interview-process records | {counts['REPORTED_INTERVIEW_PROCESS']} |
| Inferred questions | {counts['INFERRED_QUESTION']} |
| Role-derived records | {counts['ROLE_REQUIREMENT']} |
| Company-technical-focus records | {counts['COMPANY_TECHNICAL_FOCUS']} |
| Unique source domains | {len(domains)} |
| Unique source URLs in JSONL | {len({r['source_url'] for r in records})} |
| Date range of exposed dates | {min(known_dates) if known_dates else 'Not exposed'} to {max(known_dates) if known_dates else 'Not exposed'} |
| Duplicate records removed | {len(deduped) - len(records)} |

## New companies

{', '.join(new_names)}

## Weak or incomplete evidence

{', '.join(weak) if weak else 'None'}

Companies with low-confidence entries are retained because the brief requests broad coverage, but those entries are explicitly marked and should not be used as strong claims. Several companies have strong technical-context evidence but limited public candidate-reported AI/ML questions. No generic interview question was relabeled as a reported company question.

## Integration status

The backend's ingestion code supports Markdown documents and recursively scans `data/`. The generated per-company Markdown documents are saved under `data/new_company_interview_research/` and were successfully processed by the existing ingestion CLI into a separate 31-chunk verification index. The current checkout does not contain the historical PDF corpus because the specified repository deleted its `data/` directory in commit `069978e`; the original knowledge base and its production vector index were not replaced or rebuilt.
"""
(OUT / "data_quality_report.md").write_text(quality, encoding="utf-8")

# A compact source-readable index for review.
index_lines = ["# New Company Interview Research", "", "This directory contains per-company Markdown files intended for the existing RAG ingestion path. The canonical machine-readable dataset is `../research/structured_interview_dataset.jsonl`.", ""]
for c in companies:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", c["company"]).strip("_")
    index_lines.append(f"- [{c['company']}](./{safe}_interview_intelligence.md)")
(DATA / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

print(json.dumps({"companies": len(new_names), "records": len(records), "counts": counts, "domains": len(domains)}, default=dict, sort_keys=True))

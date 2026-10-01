"""Verify that the 20 queries in benchmark_dataset.json match the user request verbatim."""

import json
from pathlib import Path

prompt_queries = {
    "Q01": "What behavioral interview questions are commonly asked at Amazon, and which Leadership Principles do they test?",
    "Q02": "I'm preparing for an Amazon software engineering interview. Give me behavioral questions I should practice, and explain what a strong answer should demonstrate for each.",
    "Q03": "What technical topics should I prepare for a Google software engineering interview, particularly around data structures and algorithms?",
    "Q04": "For a Google software engineering interview, what kinds of coding problems and problem-solving skills should I expect across the technical rounds?",
    "Q05": "Compare Amazon and Google software engineering interviews in terms of their behavioral and technical interview expectations.",
    "Q06": "What does Amazon's Leadership Principle \"Dive Deep\" mean in an interview context, and what type of behavioral question could be used to evaluate it?",
    "Q07": "I keep seeing questions about ownership in Amazon interviews. What interview questions are associated with this theme, and what evidence from Amazon's interview guidance supports that?",
    "Q08": "Suppose I am asked a behavioral question about a project that failed during an Amazon interview. Which Leadership Principles could be relevant, and how should I structure my response?",
    "Q09": "What are the main interview rounds for an Amazon software engineering candidate, and what should I prepare for in each round?",
    "Q10": "I'm interviewing for an AI/ML role at NVIDIA. What technical areas and interview topics should I prioritize?",
    "Q11": "What kinds of GPU, CUDA, or deep-learning questions might be relevant when preparing for an NVIDIA technical interview?",
    "Q12": "What behavioral interview topics should I prepare for at NVIDIA, and how do they differ from the technical topics I should study?",
    "Q13": "If an interviewer asks me to describe a time I disagreed with a teammate, what company-specific interview expectations should I consider when answering for Amazon?",
    "Q14": "Which Amazon interview questions would require me to discuss measurable impact, and what Leadership Principles are most closely connected to them?",
    "Q15": "I'm preparing for interviews at multiple companies. How should my preparation differ between Amazon's behavioral interviews and NVIDIA's technical interviews?",
    "Q16": "What interview questions should I practice if I want to demonstrate problem solving, technical depth, and ownership rather than just memorizing coding patterns?",
    "Q17": "Give me examples of technical interview questions where the exact terminology matters, and explain what concepts each question is testing.",
    "Q18": "If the available interview material does not contain enough evidence to answer a company-specific question, how should the system respond instead of making up an answer?",
    "Q19": "What is the difference between preparing for behavioral questions based on company principles and preparing for technical coding questions? Use the interview material in the knowledge base to explain.",
    "Q20": "I have an upcoming software engineering interview and only one week to prepare. Based on the company's interview material in the knowledge base, what should I prioritize across behavioral, technical, and interview-process preparation?"
}

dataset_path = Path("evaluation/jev_20/benchmark_dataset.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} items from {dataset_path}")
all_match = True
for item in dataset:
    qid = item["id"]
    q = item["query"]
    expected = prompt_queries.get(qid)
    if q != expected:
        print(f"MISMATCH for {qid}:")
        print(f"  Dataset:  {repr(q)}")
        print(f"  Expected: {repr(expected)}")
        all_match = False
    else:
        print(f"[OK] {qid} verbatim match verified.")

if all_match and len(dataset) == 20:
    print("\nSUCCESS: All 20 queries are present, in order, and 100% VERBATIM MATCH!")
else:
    print("\nFAILURE: Mismatch found!")

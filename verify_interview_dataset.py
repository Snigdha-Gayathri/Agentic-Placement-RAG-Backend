from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from backend.app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent
jsonl = ROOT / "research" / "structured_interview_dataset.jsonl"
rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
required = {"company", "role", "evidence_type", "question", "interview_round", "interview_date", "source", "source_title", "source_url", "source_date", "technical_domain", "evidence", "confidence", "notes"}
allowed = {"REPORTED_QUESTION", "REPORTED_INTERVIEW_PROCESS", "ROLE_REQUIREMENT", "COMPANY_TECHNICAL_FOCUS", "INFERRED_QUESTION"}
existing = {"Accenture", "Adobe", "Amazon", "Apple", "Capgemini", "Cognizant", "Directi", "Expedia", "Meta", "Flipkart", "Goldman Sachs", "Google", "IBM", "Infosys", "JP Morgan", "LTIMindtree", "LinkedIn", "Microsoft", "NVIDIA", "Netflix", "Oracle", "TCS", "Twitter", "Uber", "Visa", "VMware", "Walmart", "Zoho", "Deloitte"}
new_companies = {r["company"] for r in rows}
assert len(new_companies) == 30, len(new_companies)
assert not existing.intersection(new_companies), existing.intersection(new_companies)
assert all(required.issubset(r) for r in rows)
assert all(r["evidence_type"] in allowed for r in rows)
assert all(r["source_url"] for r in rows)
assert all(r["technical_domain"] for r in rows)
assert all(r["evidence_type"] != "REPORTED_QUESTION" or r["question"] for r in rows)
assert all(r["evidence_type"] != "INFERRED_QUESTION" or "inferred" in r["notes"].lower() for r in rows)
keys = [(r["company"], r["evidence_type"], re.sub(r"\s+", " ", r["question"]).strip().lower()) for r in rows if r["question"]]
assert len(keys) == len(set(keys)), "duplicate company/type/question keys"
forbidden = {"Amazon", "Microsoft", "Google", "NVIDIA", "Meta", "Adobe", "Walmart", "Accenture", "Deloitte", "IBM"}
assert not forbidden.intersection(new_companies)

index = ROOT / "research" / "new_company_vector_index.json"
store = VectorStore(index_path=str(index), documents_path=str(ROOT / "data" / "new_company_interview_research"), chunk_size=500, chunk_overlap=50)
store.load()
assert store.is_ready()
results = store.search("RAG agent evaluation inference optimization interview", top_k=5)
assert len(results) == 5

print(json.dumps({"new_companies": len(new_companies), "records": len(rows), "evidence_counts": Counter(r["evidence_type"] for r in rows), "index_ready": store.is_ready(), "sample_results": [{"source": x.source_path, "score": round(x.score, 4)} for x in results]}, default=dict, sort_keys=True))

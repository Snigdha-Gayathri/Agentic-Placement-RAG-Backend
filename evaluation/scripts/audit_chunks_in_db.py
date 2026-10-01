"""Audit chunk existence and grounding for the 20 benchmark queries."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vector_store.chroma import ChromaVectorStore

def audit():
    store = ChromaVectorStore()
    coll = store._collection
    all_data = coll.get()
    all_ids = set(all_data["ids"])
    print(f"Total chunks in vector store: {len(all_ids)}")
    
    with open("evaluation/jev_20/benchmark_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    missing_chunks = {}
    valid_chunks_count = 0
    total_chunks_expected = 0
    
    for item in dataset:
        qid = item["id"]
        cids = item.get("relevant_chunk_ids", [])
        total_chunks_expected += len(cids)
        missing = [cid for cid in cids if cid not in all_ids]
        if missing:
            missing_chunks[qid] = missing
        valid_chunks_count += (len(cids) - len(missing))
        
    print(f"Total relevant chunk assignments across 20 queries: {total_chunks_expected}")
    print(f"Valid chunk IDs found in Chroma: {valid_chunks_count}")
    if missing_chunks:
        print(f"Missing chunk IDs: {missing_chunks}")
    else:
        print("All relevant chunk IDs exist in the Chroma vector store!")

if __name__ == "__main__":
    audit()

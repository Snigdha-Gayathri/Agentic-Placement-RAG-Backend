"""Query router to determine optimal retrieval strategy and metadata filters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingDecision:
    """Decision produced by the QueryRouter."""

    retriever_type: str = "hybrid"
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    needs_multi_hop: bool = False
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever_type": self.retriever_type,
            "metadata_filters": self.metadata_filters,
            "needs_multi_hop": self.needs_multi_hop,
            "reasoning": self.reasoning,
        }


class QueryRouter:
    """Heuristic and pattern-based query router for selecting retrieval strategy and filters."""

    COMPANIES = {
        "amazon",
        "google",
        "meta",
        "facebook",
        "microsoft",
        "apple",
        "adobe",
        "goldman sachs",
        "jp morgan",
        "linkedin",
        "oracle",
        "walmart",
        "twitter",
        "uber",
        "vmware",
        "visa",
        "directi",
        "expedia",
        "netflix",
        "flipkart",
        "zoho",
        "nvidia",
        "tcs",
        "infosys",
        "capgemini",
        "cognizant",
        "accenture",
        "ltimindtree",
        "ibm",
    }

    TOPICS = {
        "dsa",
        "sql",
        "system design",
        "behavioral",
        "hr",
        "dynamic programming",
        "graphs",
        "trees",
        "algorithms",
        "data structures",
        "machine learning",
        "ml",
        "deep learning",
        "ai",
        "arrays",
        "strings",
        "linked list",
        "stack",
        "queue",
        "heap",
        "hash table",
        "sorting",
        "searching",
        "recursion",
        "backtracking",
        "greedy",
        "bit manipulation",
        "math",
        "os",
        "operating system",
        "database",
        "networking",
        "security",
        "cloud",
        "distributed systems",
        "scalability",
    }

    CANONICAL_COMPANIES = {
        "amazon": "Amazon",
        "google": "Google",
        "meta": "Meta",
        "facebook": "Meta",
        "microsoft": "Microsoft",
        "apple": "Apple",
        "adobe": "Adobe",
        "goldman sachs": "Goldman Sachs",
        "jp morgan": "JP Morgan",
        "linkedin": "LinkedIn",
        "oracle": "Oracle",
        "walmart": "Walmart",
        "twitter": "Twitter",
        "uber": "Uber",
        "vmware": "VMware",
        "visa": "Visa",
        "directi": "Directi",
        "expedia": "Expedia",
        "netflix": "Netflix",
        "flipkart": "Flipkart",
        "zoho": "Zoho",
        "nvidia": "NVIDIA",
        "tcs": "TCS",
        "infosys": "Infosys",
        "capgemini": "Capgemini",
        "cognizant": "Cognizant",
        "accenture": "Accenture",
        "ltimindtree": "LTIMindtree",
        "ibm": "IBM",
    }

    def route(self, query: str) -> RoutingDecision:
        lower = query.lower()
        filters: dict[str, Any] = {}
        reasoning_parts: list[str] = []

        # Detect company
        for comp, canonical in self.CANONICAL_COMPANIES.items():
            if re.search(r"\b" + re.escape(comp) + r"\b", lower):
                filters["company"] = canonical
                reasoning_parts.append(f"Targeting company: {canonical}")
                break

        # Detect topic
        for topic in self.TOPICS:
            if re.search(r"\b" + re.escape(topic) + r"\b", lower):
                filters["topic"] = topic.replace(" ", "_")
                reasoning_parts.append(f"Targeting topic: {topic}")
                break

        # Decide retriever type
        # Exact code / problem name search favors BM25 or Hybrid; conceptual favors Dense/Hybrid
        retriever_type = "hybrid"
        if re.search(r"exact\s+problem|problem\s+#\d+|leetcode\s+\d+", lower):
            retriever_type = "bm25"
            reasoning_parts.append("Exact identifier query -> chose BM25")
        elif "explain concept" in lower or "what is the difference" in lower:
            retriever_type = "dense"
            reasoning_parts.append("Conceptual query -> chose Dense")
        else:
            reasoning_parts.append("General technical interview query -> chose Hybrid RRF")

        # Multi-hop detection
        needs_multi_hop = bool(
            "compare" in lower
            and ("and" in lower or "vs" in lower)
            or "difference between" in lower
            or len(query.split()) > 25
        )
        if needs_multi_hop:
            reasoning_parts.append("Complex comparison or long query -> multi-hop enabled")

        reasoning = "; ".join(reasoning_parts) or "Default hybrid retrieval strategy"
        return RoutingDecision(
            retriever_type=retriever_type,
            metadata_filters=filters,
            needs_multi_hop=needs_multi_hop,
            reasoning=reasoning,
        )

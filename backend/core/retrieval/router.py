"""Query router to determine optimal retrieval strategy, metadata filters, and LLM bypass decisions."""

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
    route_type: str = "SINGLE_HOP"
    llm_required: bool = True
    llm_bypass_reason: str = ""
    direct_response: str = ""
    detected_companies: list[str] = field(default_factory=list)
    detected_topics: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever_type": self.retriever_type,
            "metadata_filters": self.metadata_filters,
            "needs_multi_hop": self.needs_multi_hop,
            "route_type": self.route_type,
            "llm_required": self.llm_required,
            "llm_bypass_reason": self.llm_bypass_reason,
            "direct_response": self.direct_response,
            "detected_companies": self.detected_companies,
            "detected_topics": self.detected_topics,
            "reasoning": self.reasoning,
        }


class QueryRouter:
    """Fast deterministic pattern and entity-based query router."""

    COMPANIES = {
        "amazon", "google", "meta", "facebook", "microsoft", "apple", "adobe",
        "goldman sachs", "jp morgan", "linkedin", "oracle", "walmart", "twitter",
        "uber", "vmware", "visa", "directi", "expedia", "netflix", "flipkart",
        "zoho", "nvidia", "tcs", "infosys", "capgemini", "cognizant", "accenture",
        "ltimindtree", "ibm", "openai", "anthropic", "cohere", "perplexity",
        "xai", "databricks", "snowflake", "scale ai", "langchain", "replit",
        "weights & biases", "harvey", "runway", "glean", "together ai", "groq",
        "salesforce", "servicenow", "palantir", "stripe", "atlassian", "amd",
        "intel", "qualcomm", "paypal", "mastercard", "spotify", "bytedance",
        "tencent", "alibaba",
    }

    TOPICS = {
        "dsa", "sql", "system design", "behavioral", "hr", "dynamic programming",
        "graphs", "trees", "algorithms", "data structures", "machine learning",
        "ml", "deep learning", "ai", "llm", "rag", "agent", "agents",
        "evaluation", "reliability", "inference optimization", "ai safety",
        "fine tuning", "arrays", "strings", "linked list", "stack", "queue",
        "heap", "hash table", "sorting", "searching", "recursion", "backtracking",
        "greedy", "bit manipulation", "math", "os", "operating system",
        "database", "networking", "security", "cloud", "distributed systems",
        "scalability",
    }

    CANONICAL_COMPANIES = {
        "amazon": "Amazon", "google": "Google", "meta": "Meta", "facebook": "Meta",
        "microsoft": "Microsoft", "apple": "Apple", "adobe": "Adobe",
        "goldman sachs": "Goldman Sachs", "jp morgan": "JP Morgan",
        "linkedin": "LinkedIn", "oracle": "Oracle", "walmart": "Walmart",
        "twitter": "Twitter", "uber": "Uber", "vmware": "VMware", "visa": "Visa",
        "directi": "Directi", "expedia": "Expedia", "netflix": "Netflix",
        "flipkart": "Flipkart", "zoho": "Zoho", "nvidia": "NVIDIA", "tcs": "TCS",
        "infosys": "Infosys", "capgemini": "Capgemini", "cognizant": "Cognizant",
        "accenture": "Accenture", "ltimindtree": "LTIMindtree", "ibm": "IBM",
        "openai": "OpenAI", "anthropic": "Anthropic", "cohere": "Cohere",
        "perplexity": "Perplexity", "perplexity ai": "Perplexity", "xai": "xAI",
        "x.ai": "xAI", "databricks": "Databricks", "snowflake": "Snowflake",
        "scale ai": "Scale AI", "scale": "Scale AI", "langchain": "LangChain",
        "replit": "Replit", "weights & biases": "Weights & Biases",
        "weights and biases": "Weights & Biases", "wandb": "Weights & Biases",
        "harvey": "Harvey", "runway": "Runway", "glean": "Glean",
        "together ai": "Together AI", "together": "Together AI", "together.ai": "Together AI",
        "groq": "Groq", "salesforce": "Salesforce", "servicenow": "ServiceNow",
        "palantir": "Palantir", "stripe": "Stripe", "atlassian": "Atlassian",
        "amd": "AMD", "intel": "Intel", "qualcomm": "Qualcomm", "paypal": "PayPal",
        "mastercard": "Mastercard", "spotify": "Spotify", "bytedance": "ByteDance",
        "tencent": "Tencent", "alibaba": "Alibaba",
    }

    OUT_OF_DOMAIN_PATTERNS = [
        r"\b(weather|forecast|temperature in|rain today)\b",
        r"\b(recipe|cook|baking|ingredients for)\b",
        r"\b(movie|cinema|actor|box office)\b",
        r"\b(football|cricket|nba|fifa|score of the match)\b",
        r"\b(horoscope|astrology|zodiac)\b",
        r"\b(gpa|student ranking|placement package|highest ctc|average package)\b",
    ]

    def route(self, query: str) -> RoutingDecision:
        lower = query.lower().strip()
        filters: dict[str, Any] = {}
        reasoning_parts: list[str] = []

        # 1. Check for Out-of-Domain (Strict Abstention without LLM call)
        for pattern in self.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, lower):
                return RoutingDecision(
                    retriever_type="none",
                    metadata_filters={},
                    needs_multi_hop=False,
                    route_type="OUT_OF_DOMAIN",
                    llm_required=False,
                    llm_bypass_reason="Query is outside technical interview preparation domain.",
                    direct_response=(
                        "I am an evidence-grounded interview preparation assistant. "
                        "The query is outside the scope of technical interview intelligence, "
                        "which focuses exclusively on company interview processes, DSA, system design, "
                        "and technical interview questions."
                    ),
                    reasoning="Out-of-domain pattern detected -> Bypassed LLM generation with direct abstention.",
                )

        # 2. Check for Direct Metadata Queries (Instant deterministic answer with 0 LLM calls)
        if re.search(r"\b(what|which|list|show|give me|tell me)\b.*\b(companies|tech companies|organizations)\b", lower) or lower in [
            "list companies", "supported companies", "indexed companies", "companies in db",
            "companies in database", "what companies are in the database?", "what companies are indexed?",
            "what companies are indexed in the knowledge base?"
        ]:
            sorted_comps = sorted(set(self.CANONICAL_COMPANIES.values()))
            comp_list_md = ", ".join(sorted_comps)
            return RoutingDecision(
                retriever_type="none",
                metadata_filters={},
                needs_multi_hop=False,
                route_type="METADATA_DIRECT",
                llm_required=False,
                llm_bypass_reason="Direct metadata lookup of indexed companies resolved deterministically.",
                direct_response=(
                    f"### Indexed Company Knowledge Base ({len(sorted_comps)} Companies)\n\n"
                    f"The interview knowledge base currently indexes verified interview research, reported questions, and technical preparation handbooks for the following **{len(sorted_comps)} companies**:\n\n"
                    f"{comp_list_md}\n\n"
                    "You can ask company-specific interview questions (e.g. *'What technical rounds does OpenAI conduct?'*), topic questions (*'DSA dynamic programming questions'*), or multi-company comparisons (*'Compare OpenAI vs Anthropic interview focus'*)."
                ),
                reasoning="Metadata inventory query -> Deterministically answered with 0 LLM calls.",
            )

        if re.search(r"\b(what|which|list|show|give me|tell me)\b.*\b(topics|interview topics|domains)\b", lower) or lower in [
            "list topics", "supported topics", "indexed topics", "list all interview topics"
        ]:
            sorted_topics = sorted({t.title() for t in self.TOPICS})
            topics_md = ", ".join(sorted_topics[:25])
            return RoutingDecision(
                retriever_type="none",
                metadata_filters={},
                needs_multi_hop=False,
                route_type="METADATA_DIRECT",
                llm_required=False,
                llm_bypass_reason="Direct metadata lookup of covered interview topics resolved deterministically.",
                direct_response=(
                    "### Covered Technical Interview Topics\n\n"
                    f"The knowledge corpus contains interview questions and preparation guidance across key domains including:\n\n"
                    f"- **Data Structures & Algorithms**: {topics_md}\n"
                    "- **Systems & Architecture**: System Design, Distributed Systems, Scalability, OS, Databases, Networking\n"
                    "- **AI & Machine Learning**: LLM Agents, RAG Pipelines, Inference Optimization, AI Safety, RLHF, Fine-Tuning\n"
                    "- **Behavioral & Leadership**: Company Leadership Principles, Behavioral Rounds, HR Rounds\n"
                ),
                reasoning="Topic metadata query -> Deterministically answered with 0 LLM calls.",
            )

        # 3. Detect companies mentioned in query
        detected_companies: list[str] = []
        for comp, canonical in sorted(self.CANONICAL_COMPANIES.items(), key=lambda x: -len(x[0])):
            pattern = r"\b" + re.escape(comp) + r"\b"
            if re.search(pattern, lower):
                if canonical not in detected_companies:
                    detected_companies.append(canonical)

        # 4. Detect topics mentioned in query
        detected_topics: list[str] = []
        for topic in self.TOPICS:
            if re.search(r"\b" + re.escape(topic) + r"\b", lower):
                topic_key = topic.replace(" ", "_")
                if topic_key not in detected_topics:
                    detected_topics.append(topic_key)

        # Multi-company or comparison detection
        is_comparison = bool(
            len(detected_companies) > 1
            or ("compare" in lower and ("and" in lower or "vs" in lower))
            or "difference between" in lower
            or " vs " in lower
        )

        # Multi-round detection (multi-hop within a single company)
        has_multi_round = bool(
            ("round" in lower and ("1" in lower or "2" in lower or "technical" in lower or "hr" in lower or "all rounds" in lower or "multiple rounds" in lower))
            or "interview stages" in lower
            or "interview process and coding" in lower
        )

        needs_multi_hop = is_comparison or has_multi_round or len(query.split()) > 25

        if len(detected_companies) == 1 and not is_comparison:
            filters["company"] = detected_companies[0]
            reasoning_parts.append(f"Single target company detected: {detected_companies[0]} -> filtered retrieval")
            route_type = "SINGLE_HOP"
        elif len(detected_companies) > 1 or is_comparison:
            reasoning_parts.append(
                f"Multi-company comparison detected ({', '.join(detected_companies) if detected_companies else 'multiple'}) -> multi-hop hybrid retrieval across sources"
            )
            route_type = "MULTI_HOP"
        elif detected_topics:
            reasoning_parts.append(f"Topic-specific query ({', '.join(detected_topics)}) -> hybrid retrieval")
            route_type = "CONCEPTUAL"
        else:
            reasoning_parts.append("General technical interview query -> hybrid retrieval")
            route_type = "SINGLE_HOP"

        # Determine retrieval method
        retriever_type = "hybrid"
        if re.search(r"exact\s+problem|problem\s+#\d+|leetcode\s+\d+", lower):
            retriever_type = "bm25"
            reasoning_parts.append("Exact identifier query -> Lexical BM25 prioritized")
        elif "explain concept" in lower or "what is the difference" in lower:
            retriever_type = "dense"
            reasoning_parts.append("Conceptual query -> Dense semantic search prioritized")
        else:
            retriever_type = "hybrid"
            reasoning_parts.append("Hybrid Reciprocal Rank Fusion (Dense + BM25)")

        if needs_multi_hop:
            reasoning_parts.append("Multi-hop reasoning enabled to gather evidence across rounds/entities")

        reasoning = "; ".join(reasoning_parts)

        return RoutingDecision(
            retriever_type=retriever_type,
            metadata_filters=filters,
            needs_multi_hop=needs_multi_hop,
            route_type=route_type,
            llm_required=True,
            llm_bypass_reason="",
            detected_companies=detected_companies,
            detected_topics=detected_topics,
            reasoning=reasoning,
        )

"""Follow-up query rewriter resolving pronouns and conversation references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conversation import Turn


@dataclass
class RewriteResult:
    """Result of context-aware query rewriting."""

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "was_rewritten": self.was_rewritten,
            "reasoning": self.reasoning,
        }


class QueryRewriter:
    """Heuristic and LLM-assisted query rewriter for conversational follow-ups."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    async def rewrite(self, query: str, history: list[Turn]) -> RewriteResult:
        if not history or len(history) < 2:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                was_rewritten=False,
                reasoning="Standalone query; no dialogue history present.",
            )

        lower = query.lower()
        pronouns = {"it", "this", "that", "they", "them", "these", "those", "he", "she"}
        query_words = set(lower.split())
        has_pronoun = bool(query_words.intersection(pronouns))

        # Check if query is short follow-up like "what about google?" or "how about the time complexity?"
        is_followup = bool(
            has_pronoun
            or lower.startswith("what about")
            or lower.startswith("how about")
            or lower.startswith("and ")
            or len(query.split()) <= 4
        )

        if not is_followup:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                was_rewritten=False,
                reasoning="Query appears self-contained without conversational references.",
            )

        last_user = next((t.content for t in reversed(history) if t.role == "user"), "")
        last_asst = next((t.content for t in reversed(history) if t.role == "assistant"), "")

        if self.llm:
            try:
                system_p = (
                    "Rewrite the follow-up question to be completely standalone and clear, "
                    "incorporating necessary context from previous turns. Output ONLY the rewritten question."
                )
                user_p = f"Previous user question: {last_user}\nPrevious answer excerpt: {last_asst[:300]}\nFollow-up question: {query}"
                resp = await self.llm.generate(system_prompt=system_p, user_prompt=user_p)
                rewritten = resp.text.strip()
                if rewritten and len(rewritten) > 3:
                    return RewriteResult(
                        original_query=query,
                        rewritten_query=rewritten,
                        was_rewritten=True,
                        reasoning="Resolved conversational pronouns/references using LLM context.",
                    )
            except Exception:
                pass

        # Heuristic fallback rewrite
        rewritten = f"{last_user} — follow up: {query}"
        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            was_rewritten=True,
            reasoning="Heuristic rewrite combining previous user turn with current follow-up.",
        )

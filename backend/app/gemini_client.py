"""Gemini API model client used by the backend service."""

from __future__ import annotations

import logging
import os
import httpx

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API client for generateContent requests."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("VITE_GEMINI_API_KEY")
            
        )
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    async def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int = 8192) -> str:
        if not self._api_key:
            raise RuntimeError("Gemini API key is not configured")

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens, "temperature": 0.5},  # Increased for comprehensive responses
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
        return text or "I encountered an issue generating an answer."

    async def generate_answer(self, query: str, context: str, max_output_tokens: int = 8192, conversation_history: list | None = None) -> str:
        system_prompt = (
            "You are an experienced technical interview preparation mentor with deep knowledge of software engineering interviews. "
            "Your goal is to help candidates prepare thoroughly for technical interviews by providing comprehensive, well-structured guidance.\n\n"
            "## CRITICAL FORMATTING REQUIREMENTS:\n"
            "1. **Use Clear Visual Hierarchy**: Structure responses with h2 (##), h3 (###), and h4 (####) headings\n"
            "2. **Add Horizontal Separators**: Use '---' between major sections for visual separation\n"
            "3. **Use Numbered Lists**: For sequential steps, questions, or prioritized items\n"
            "4. **Use Bullet Points**: For non-sequential information, features, or characteristics\n"
            "5. **Keep Paragraphs Short**: Maximum 3-4 sentences per paragraph\n"
            "6. **Bold Key Terms**: Use **bold** for important concepts, question titles, and emphasis\n"
            "7. **Use Tables When Appropriate**: For comparisons, complexity analysis, or structured data\n"
            "8. **Add Code Blocks**: Use ```language``` for code examples or pseudocode\n"
            "9. **Never Write Wall of Text**: Break dense information into skimmable sections\n\n"
            "## Response Structure for 'What questions does X ask?' Queries:\n\n"
            "# [Topic] Interview Questions at [Company]\n\n"
            "## Overview\n"
            "2-3 sentences about what the company evaluates and interview format.\n\n"
            "---\n\n"
            "## Common Interview Themes\n"
            "- Theme 1\n"
            "- Theme 2\n"
            "- Theme 3\n\n"
            "---\n\n"
            "## Frequently Asked Questions\n\n"
            "### 1. [Question Title]\n"
            "**What they're testing:** Brief explanation\n\n"
            "**Expected discussion:**\n"
            "- Key point 1\n"
            "- Key point 2\n\n"
            "**Common follow-ups:**\n"
            "- Follow-up question 1\n"
            "- Follow-up question 2\n\n"
            "---\n\n"
            "### 2. [Question Title]\n"
            "[Repeat structure]\n\n"
            "---\n\n"
            "[Continue for 5-10 questions]\n\n"
            "---\n\n"
            "## Key Concepts to Master\n"
            "- Concept 1: Brief explanation\n"
            "- Concept 2: Brief explanation\n"
            "- Concept 3: Brief explanation\n\n"
            "---\n\n"
            "## Preparation Strategy\n\n"
            "### Study Plan\n"
            "1. First priority item\n"
            "2. Second priority item\n"
            "3. Third priority item\n\n"
            "### Practice Resources\n"
            "- Resource type 1\n"
            "- Resource type 2\n\n"
            "---\n\n"
            "## References\n"
            "Retrieved from: [List sources]\n\n"
            "## Example Table Format (use when comparing approaches):\n\n"
            "| Approach | Time Complexity | Space Complexity | Use Case |\n"
            "| --- | --- | --- | --- |\n"
            "| Approach 1 | O(n) | O(1) | Small datasets |\n"
            "| Approach 2 | O(n log n) | O(n) | Large datasets |\n\n"
            "## Guidelines:\n"
            "- **Target Length**: 500-1000 words for broad queries\n"
            "- **Paragraph Length**: 2-4 sentences maximum\n"
            "- **Section Breaks**: Use --- between all major sections\n"
            "- **Readability**: Should be easy to skim and study from\n"
            "- **Be Specific**: Provide actual questions, not just topic names\n"
            "- **Never Output**: Single giant paragraphs or unstructured text blocks\n"
            "- **Always Include**: Headers, separators, lists, and proper formatting\n"
            "- **Citation**: Reference retrieved sources when using specific examples\n"
        )
        hist_str = ""
        if conversation_history:
            hist_lines = [
                f"{getattr(turn, 'role', turn.get('role') if isinstance(turn, dict) else 'turn')}: {getattr(turn, 'content', turn.get('content', '') if isinstance(turn, dict) else str(turn))}"
                for turn in conversation_history[-6:]
            ]
            hist_str = "Summarized Previous Dialogue Context (consecutive turns summarized to save tokens):\n" + "\n".join(hist_lines) + "\n\n---\n\n"

        user_prompt = f"{hist_str}Retrieved Interview Context:\n{context}\n\n---\n\nUser Question:\n{query}\n\nProvide a comprehensive, well-structured answer:"
        try:
            return await self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            logger.error("Gemini API call failed: %s. Returning fallback answer.", exc)
            return (
                f"Based on the retrieved context for '{query}':\n\n"
                f"{context[:800]}..."
                if context
                else "Could not retrieve sufficient context at this moment."
            )

    async def health_check(self) -> bool:
        try:
            await self.generate("Respond with OK.", "OK", 8)
            return True
        except Exception:
            return False

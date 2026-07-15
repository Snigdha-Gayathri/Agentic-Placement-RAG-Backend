"""Metadata extraction from document filenames and content patterns."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Extracted metadata for a single document."""

    company: str = "General"
    topic: str = "dsa"
    difficulty: str = "mixed"
    source_type: str = "leetcode"
    tags: list[str] = field(default_factory=list)
    page_count: int = 0
    filename: str = ""


class MetadataExtractor:
    """Extract company, topic, and other metadata from document filenames and content.

    Uses pattern matching tuned for the specific placement preparation PDF corpus
    in the data/ directory. Company names are normalized (e.g., Facebook → Meta).
    """

    # Company name normalizations (display name mapping)
    COMPANY_ALIASES: ClassVar[dict[str, str]] = {
        "facebook": "Meta",
        "fb": "Meta",
        "meta": "Meta",
        "amazon": "Amazon",
        "google": "Google",
        "microsoft": "Microsoft",
        "apple": "Apple",
        "adobe": "Adobe",
        "oracle": "Oracle",
        "twitter": "Twitter",
        "uber": "Uber",
        "linkedin": "LinkedIn",
        "visa": "Visa",
        "vmware": "VMware",
        "walmart": "Walmart",
        "expedia": "Expedia",
        "directi": "Directi",
        "goldman sachs": "Goldman Sachs",
        "goldmansachs": "Goldman Sachs",
        "jp morgan": "JP Morgan",
        "jpmorgan": "JP Morgan",
    }

    # Ordered patterns for company extraction from filenames.
    # More specific patterns come first to avoid partial matches.
    COMPANY_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"goldman\s*sachs", "Goldman Sachs"),
        (r"jp\s*morgan", "JP Morgan"),
        (r"walmart\s*labs?", "Walmart"),
        (r"walmart", "Walmart"),
        (r"facebook|𝐅𝐁", "Meta"),
        (r"amazon|𝐀𝐌𝐀𝐙𝐎𝐍|𝐀𝐦𝐚𝐳𝐨𝐧", "Amazon"),
        (r"google", "Google"),
        (r"microsoft", "Microsoft"),
        (r"apple", "Apple"),
        (r"adobe", "Adobe"),
        (r"oracle", "Oracle"),
        (r"twitter", "Twitter"),
        (r"uber", "Uber"),
        (r"linkedin", "LinkedIn"),
        (r"visa\b", "Visa"),
        (r"vmware", "VMware"),
        (r"expedia", "Expedia"),
        (r"directi", "Directi"),
    ]

    # Topic classification patterns
    TOPIC_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"\bsql\b", "sql"),
        (r"system\s*design", "system_design"),
        (r"behavioral|leadership|lp", "behavioral"),
        (r"\bdsa\b|data\s*structure|algorithm|stacks|queues|linked\s*list|tree|graph|sort|search|dynamic\s*programming", "dsa"),
        (r"interview\s*guide|prepare|preparation|revision", "interview_prep"),
        (r"leetcode|leet\s*code", "dsa"),
        (r"tagged", "dsa"),
    ]

    # Difficulty heuristics from content keywords
    DIFFICULTY_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "easy": ["easy", "basic", "simple", "beginner", "introduct"],
        "medium": ["medium", "intermediate", "moderate"],
        "hard": ["hard", "difficult", "advanced", "complex", "challenging"],
    }

    def extract(
        self,
        filename: str,
        text: str = "",
        page_count: int = 0,
    ) -> DocumentMetadata:
        """Extract metadata from a document filename and optional content.

        Args:
            filename: The document filename (e.g., 'Amazon_DSA.pdf').
            text: Optional extracted text content for deeper analysis.
            page_count: Number of pages in the document.

        Returns:
            DocumentMetadata with inferred company, topic, tags, etc.
        """
        normalized_name = self._normalize_unicode(filename)
        company = self._extract_company(normalized_name)
        topic = self._extract_topic(normalized_name, text)
        difficulty = self._extract_difficulty(text) if text else "mixed"
        source_type = self._extract_source_type(normalized_name)
        tags = self._extract_tags(normalized_name, text, company, topic)

        return DocumentMetadata(
            company=company,
            topic=topic,
            difficulty=difficulty,
            source_type=source_type,
            tags=tags,
            page_count=page_count,
            filename=filename,
        )

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode fancy text (bold/italic math chars) to ASCII equivalents."""
        # Map mathematical bold/italic Unicode ranges to ASCII
        result: list[str] = []
        for char in text:
            code = ord(char)
            # Mathematical Bold Capital A-Z: U+1D400 - U+1D419
            if 0x1D400 <= code <= 0x1D419:
                result.append(chr(code - 0x1D400 + ord("A")))
            # Mathematical Bold Small a-z: U+1D41A - U+1D433
            elif 0x1D41A <= code <= 0x1D433:
                result.append(chr(code - 0x1D41A + ord("a")))
            # Mathematical Bold Italic Capital A-Z: U+1D468 - U+1D481
            elif 0x1D468 <= code <= 0x1D481:
                result.append(chr(code - 0x1D468 + ord("A")))
            # Mathematical Bold Italic Small a-z: U+1D482 - U+1D49B
            elif 0x1D482 <= code <= 0x1D49B:
                result.append(chr(code - 0x1D482 + ord("a")))
            # Mathematical Sans-Serif Bold Capital A-Z: U+1D5D4 - U+1D5ED
            elif 0x1D5D4 <= code <= 0x1D5ED:
                result.append(chr(code - 0x1D5D4 + ord("A")))
            # Mathematical Sans-Serif Bold Small a-z: U+1D5EE - U+1D607
            elif 0x1D5EE <= code <= 0x1D607:
                result.append(chr(code - 0x1D5EE + ord("a")))
            else:
                # Fallback: use NFKD normalization for other special chars
                normalized = unicodedata.normalize("NFKD", char)
                result.append(normalized)
        return "".join(result)

    def _extract_company(self, filename: str) -> str:
        """Infer the company name from the filename.

        Uses ordered regex patterns. Falls back to 'General' for
        documents not clearly associated with a company.
        """
        name_lower = filename.lower()

        # Check specific patterns in priority order
        for pattern, company in self.COMPANY_PATTERNS:
            if re.search(pattern, name_lower):
                return company

        # Check for 'SDE' in the name without a company prefix → check content
        if "sde" in name_lower and "amazon" not in name_lower:
            return "General"

        # Check if "how to prepare" pattern for Amazon
        if "prepare" in name_lower and "amazon" in name_lower:
            return "Amazon"

        return "General"

    def _extract_topic(self, filename: str, text: str = "") -> str:
        """Infer the document topic from filename and content."""
        combined = f"{filename} {text[:2000]}".lower()

        for pattern, topic in self.TOPIC_PATTERNS:
            if re.search(pattern, combined):
                return topic

        return "dsa"  # Default for interview prep materials

    def _extract_difficulty(self, text: str) -> str:
        """Infer overall difficulty from content keyword frequency."""
        text_lower = text[:5000].lower()
        scores: dict[str, int] = {}

        for level, keywords in self.DIFFICULTY_KEYWORDS.items():
            count = sum(text_lower.count(kw) for kw in keywords)
            scores[level] = count

        if not any(scores.values()):
            return "mixed"

        return max(scores, key=lambda k: scores[k])

    def _extract_source_type(self, filename: str) -> str:
        """Classify the document source type."""
        name_lower = filename.lower()

        if "leetcode" in name_lower or "leet code" in name_lower:
            return "leetcode"
        if "tagged" in name_lower:
            return "tagged_problems"
        if "interview" in name_lower or "guide" in name_lower:
            return "interview_guide"
        if "dsa" in name_lower or "data structure" in name_lower:
            return "dsa_sheet"
        if "revision" in name_lower or "sheet" in name_lower:
            return "revision_sheet"
        return "general"

    def _extract_tags(
        self,
        filename: str,
        text: str,
        company: str,
        topic: str,
    ) -> list[str]:
        """Build a list of searchable tags from all available metadata."""
        tags: set[str] = set()

        # Add company and topic
        tags.add(company.lower().replace(" ", "_"))
        tags.add(topic)

        name_lower = filename.lower()

        # Add format/source tags
        if "leetcode" in name_lower:
            tags.add("leetcode")
        if "tagged" in name_lower:
            tags.add("tagged")
        if "6_months" in name_lower or "6months" in name_lower:
            tags.add("recent_6_months")
        if "part" in name_lower:
            tags.add("multi_part")

        # Content-based tags
        content_preview = text[:3000].lower() if text else ""
        content_tags = {
            "array": r"\barray\b",
            "string": r"\bstring\b",
            "tree": r"\btree\b",
            "graph": r"\bgraph\b",
            "dynamic_programming": r"\bdynamic programming\b|\bdp\b",
            "linked_list": r"\blinked\s*list\b",
            "stack": r"\bstack\b",
            "queue": r"\bqueue\b",
            "hash_table": r"\bhash\s*(table|map|set)\b",
            "binary_search": r"\bbinary\s*search\b",
            "sorting": r"\bsort(ing)?\b",
            "recursion": r"\brecurs(ion|ive)\b",
            "backtracking": r"\bbacktrack(ing)?\b",
            "greedy": r"\bgreedy\b",
            "two_pointers": r"\btwo\s*pointer\b",
            "sliding_window": r"\bsliding\s*window\b",
        }
        for tag, pattern in content_tags.items():
            if re.search(pattern, content_preview):
                tags.add(tag)

        return sorted(tags)

"""Security constants and default patterns for RAG guardrails."""

from __future__ import annotations

DEFAULT_MAX_QUERY_LENGTH = 2000
DEFAULT_MAX_CONTEXT_LENGTH = 12000
DEFAULT_MAX_OUTPUT_TOKENS = 800
DEFAULT_MAX_RETRIEVED_CHUNKS = 8
DEFAULT_RATE_LIMIT = 30
DEFAULT_RATE_WINDOW_SECONDS = 60
DEFAULT_SIMILARITY_THRESHOLD = 0.12
DEFAULT_HALLUCINATION_THRESHOLD = 0.45
DEFAULT_PROMPT_INJECTION_THRESHOLD = 0.65

SAFE_FALLBACK_RESPONSE = "I couldn't find sufficient information in the indexed documents."

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+everything",
    r"act\s+as\s+(the\s+)?system",
    r"reveal\s+(your\s+)?prompt",
    r"hidden\s+instructions",
    r"show\s+chain\s+of\s+thought",
    r"developer\s+mode",
    r"\bDAN\b",
    r"jailbreak",
    r"override\s+previous\s+instructions",
    r"system\s+prompt",
    r"instruction\s+hijack",
]

RETRIEVAL_BLOCK_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"\bassistant:\b",
    r"\bsystem:\b",
    r"\bdeveloper:\b",
    r"\btool:\b",
    r"\bfunction:\b",
    r"\bprompt:\b",
    r"password",
    r"secret",
    r"api\s*key",
    r"token",
    r"private\s*key",
    r"ssh\s*key",
    r"BEGIN\s+RSA",
    r"BEGIN\s+PRIVATE\s+KEY",
    r"rm\s+-rf",
    r"curl\s+http",
]

OUTPUT_BLOCK_PATTERNS = [
    r"system\s+prompt",
    r"internal\s+instructions",
    r"api[_-]?key",
    r"password",
    r"authorization:\s*bearer",
    r"BEGIN\s+PRIVATE\s+KEY",
    r"Traceback\s+\(most\s+recent\s+call\s+last\)",
    r"[A-Za-z]:\\\\[^\n\r]+",
]

INPUT_BLOCK_PATTERNS = [
    r"<script[\s\S]*?>[\s\S]*?<\/script>",
    r"<\/?(html|body|iframe|object|embed|xml)[^>]*>",
    r"\b(select|insert|update|delete|drop|union\s+select)\b",
    r"(;|\|\||&&)\s*(rm|curl|wget|bash|sh|powershell|cmd)\b",
    r"\x00",
]

SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{35}",
    r"(?i)api[_-]?key\s*[:=]\s*[\w\-]{10,}",
    r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9\-\._~\+\/]+=*",
    r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
    r"-----BEGIN\s+[A-Z\s]+PRIVATE\s+KEY-----",
]

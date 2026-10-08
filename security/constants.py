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

class SecurityCategory:
    PROMPT_INJECTION = "PROMPT_INJECTION"
    SYSTEM_INSTRUCTION_EXFILTRATION = "SYSTEM_INSTRUCTION_EXFILTRATION"
    SECRET_EXFILTRATION = "SECRET_EXFILTRATION"
    UNAUTHORIZED_TOOL_USE = "UNAUTHORIZED_TOOL_USE"
    SECURITY_BYPASS = "SECURITY_BYPASS"
    MALICIOUS_AUTOMATION = "MALICIOUS_AUTOMATION"
    DANGEROUS_REQUEST = "DANGEROUS_REQUEST"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"
    POLICY_VIOLATION = "POLICY_VIOLATION"


class SecuritySeverity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(previous\s+)?(instructions|rules|guardrails|policies)",
    r"forget\s+everything",
    r"act\s+as\s+(the\s+)?system",
    r"treat\s+this\s+message\s+as\s+(the\s+)?developer",
    r"reveal\s+(your\s+)?prompt",
    r"hidden\s+instructions",
    r"show\s+chain\s+of\s+thought",
    r"developer\s+mode",
    r"\bDAN\b",
    r"jailbreak",
    r"override\s+(previous\s+|security\s+)?instructions",
    r"system\s+prompt",
    r"instruction\s+hijack",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"bypass\s+(security|guardrail|retrieval)\s+restrictions",
]

SYSTEM_EXFILTRATION_PATTERNS = [
    r"(reveal|show|print|tell\s+me|display|output|dump)\s+(your\s+|the\s+)?(system\s+prompt|hidden\s+prompt|internal\s+instructions|system\s+message|system\s+instructions)",
    r"what\s+is\s+your\s+(exact\s+)?(system\s+prompt|initial\s+prompt|hidden\s+instruction)",
    r"repeat\s+the\s+words\s+above\s+starting\s+with",
]

SECRET_EXFILTRATION_PATTERNS = [
    r"(reveal|show|print|display|give\s+me|tell\s+me|exfiltrate|leak|dump)\s+(the\s+|your\s+)?([a-z_0-9-]*\s+)?([a-z_0-9-]*api[_-]?key|secret|password|credential|token|\.env|environment\s+variables?)",
    r"what\s+is\s+the\s+([a-z_0-9-]*\s+)?api[_-]?key",
    r"echo\s+\$GEMINI_API_KEY",
    r"gemini_api_key",
    r"os\.environ",
]

UNAUTHORIZED_TOOL_PATTERNS = [
    r"(execute|run|call|use)\s+(the\s+)?([a-z_0-9-]*\s+)?(bash|shell|terminal|powershell|cmd|eval|exec|arbitrary\s+code|unauthorized\s+tool)",
    r"execute\s+this\s+tool",
    r"ignore\s+retrieval\s+restrictions\s+and\s+execute",
    r"cat\s+/etc/passwd",
    r"rm\s+-rf",
]

SECURITY_BYPASS_PATTERNS = [
    r"(disable|bypass|circumvent|turn\s+off|deactivate)\s+(all\s+)?(security|safety|guardrails?|filters?|checks?|rules?)",
    r"disregard\s+your\s+security\s+rules",
]

BENIGN_INJECTION_DISCUSSION_PATTERNS = [
    r"\b(what\s+is|explain|describe|define|how\s+does|how\s+to\s+prevent|how\s+can\s+we\s+defend|defend\s+against|mitigate|prevent)\b.*\b(prompt\s+injection|jailbreak|adversarial\s+attack|prompt\s+leakage)\b",
    r"\b(prompt\s+injection|jailbreaking)\b.*\b(definition|concept|examples?|mitigation|defense|vulnerability|in\s+rag)\b",
    r"\bhow\s+can\s+rag\s+systems\s+defend\s+against\s+(it|prompt\s+injection)\b",
    r"\bwhat\s+are\s+prompt\s+injection\s+attacks\b",
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
    r"\b(auth|bearer|secret|access|session|api)[_-]?token\b",
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


"""Security package for layered RAG guardrails, authentication, and access control."""

from .auth import AuthContext, Role, get_auth_context, require_admin, require_user
from .config import SecurityConfig, load_config
from .constants import SAFE_FALLBACK_RESPONSE, SecurityCategory, SecuritySeverity
from .context_sanitizer import ContextSanitizer
from .file_lock import CrossProcessLock
from .grounding import GroundingVerifier
from .hallucination_guard import HallucinationGuard
from .input_validator import InputValidator
from .logger import SecurityLogger
from .output_sanitizer import OutputSanitizer
from .output_validator import OutputValidator
from .prompt_injection import InjectionDetectionResult, PromptInjectionDetector
from .rate_limiter import SlidingWindowRateLimiter
from .retrieval_guard import RetrievalGuard, RetrievedChunk

__all__ = [
    "AuthContext",
    "Role",
    "get_auth_context",
    "require_user",
    "require_admin",
    "CrossProcessLock",
    "SecurityConfig",
    "load_config",
    "SAFE_FALLBACK_RESPONSE",
    "SecurityCategory",
    "SecuritySeverity",
    "ContextSanitizer",
    "GroundingVerifier",
    "HallucinationGuard",
    "InputValidator",
    "SecurityLogger",
    "OutputSanitizer",
    "OutputValidator",
    "InjectionDetectionResult",
    "PromptInjectionDetector",
    "SlidingWindowRateLimiter",
    "RetrievalGuard",
    "RetrievedChunk",
]


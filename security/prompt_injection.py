"""Prompt and jailbreak injection detection using heuristic scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SecurityConfig
from .constants import PROMPT_INJECTION_PATTERNS


from dataclasses import dataclass, field

from .config import SecurityConfig
from .constants import (
    BENIGN_INJECTION_DISCUSSION_PATTERNS,
    PROMPT_INJECTION_PATTERNS,
    SECRET_EXFILTRATION_PATTERNS,
    SECURITY_BYPASS_PATTERNS,
    SYSTEM_EXFILTRATION_PATTERNS,
    UNAUTHORIZED_TOOL_PATTERNS,
    SecurityCategory,
    SecuritySeverity,
)


@dataclass(frozen=True)
class InjectionDetectionResult:
    risk_score: float
    confidence: float
    reasons: list[str]
    action: str  # "allow", "warn", "block"
    category: str = SecurityCategory.PROMPT_INJECTION
    severity: str = SecuritySeverity.HIGH
    user_safe_message: str = ""
    internal_reason_code: str = "NORMAL_QUERY"
    blocked_operations: list[str] = field(default_factory=list)


def build_user_safe_security_response(
    category: str,
    severity: str,
    detection_title: str,
    why_blocked: str,
    advice: str,
) -> str:
    """Build polished user-facing security response following security standards."""
    return (
        "### 🛡️ Security Validation Notice\n\n"
        "**🛡️ Security Check: Request Blocked**\n\n"
        f"**Detection:** {detection_title}  \n"
        f"**Severity:** {severity}  \n"
        f"**Why it was blocked:** {why_blocked}  \n"
        "**Action taken:** The request was blocked before retrieval. No credentials or protected instructions were disclosed.\n\n"
        "**Retrieval:** Not executed  \n"
        "**Your data:** No protected system information was disclosed.  \n"
        f"**Guidance:** {advice}"
    )


class PromptInjectionDetector:
    """Score query risk for direct and indirect prompt injection patterns with intent separation."""

    def __init__(self, config: SecurityConfig) -> None:
        self._threshold = config.prompt_injection_threshold

    def _classify_intent(
        self, query: str, lowered: str
    ) -> tuple[bool, str, str, str, str, str, list[str]]:
        """Multi-signal intent detection combining targeted rules, verb-target co-occurrence, and override cues.

        Returns:
            (is_attack, category, severity, detection_title, why_blocked, advice, reasons)
        """
        reasons: list[str] = []

        # Target definitions
        has_credential_target = bool(
            re.search(
                r"\b(api[ _-]*keys?|gemini[ _-]*api[ _-]*keys?|gemini[ _-]*keys?|google[ _-]*api[ _-]*keys?|secret[ _-]*keys?|access[ _-]*keys?|private[ _-]*keys?|auth[ _-]*keys?|credentials?|passwords?|passcodes?|secrets?|tokens?|auth[ _-]*tokens?|bearer[ _-]*tokens?|session[ _-]*tokens?|access[ _-]*tokens?|environment\s+variables?|env\s+vars?|env\s+variables?|\.env|os\.environ|environ|aws[_-]access|jwt[_-]secret)\b",
                lowered,
            )
        )

        has_system_prompt_target = bool(
            re.search(
                r"\b(system\s+prompts?|system\s+instructions?|system\s+messages?|hidden\s+prompts?|internal\s+instructions?|initial\s+prompts?|base\s+prompts?|developer\s+instructions?|meta\s+prompts?|system\s+rules?|developer\s+prompts?)\b",
                lowered,
            )
        )

        # Extraction actions / verbs
        has_extraction_verb = bool(
            re.search(
                r"\b(reveal|show|print|display|tell|give|dump|output|echo|exfiltrate|leak|share|disclose|provide|list|return|fetch|export|extract|read|cat|inspect|expose|write\s+out|publish|unveil|divulge|hand\s+over)\b",
                lowered,
            )
        )

        # Interrogative / extraction inquiries (e.g., "what is the ... api key being used")
        has_extraction_inquiry = bool(
            re.search(
                r"\b(what\s+(is|are|was)|whats|which|do\s+you\s+have|can\s+you\s+(share|give|disclose|provide|tell|show)|tell\s+me\s+about)\b",
                lowered,
            )
        )

        # System reference context (e.g., "your", "being used", "used by", "in use", "this system", "this app", "configured")
        has_system_reference = bool(
            re.search(
                r"\b(your|the\s+system|this\s+system|this\s+app|being\s+used|used\s+by|used\s+here|in\s+use|configured|current|internal|underlying|backend)\b",
                lowered,
            )
        )

        # Instruction override cues
        has_direct_override_cmd = bool(
            re.search(
                r"\b(ignore\s+(all\s+|your\s+|any\s+|the\s+)?(previous|prior|above|system)?\s*(instructions|directives|rules|guidelines|prompts?)|disregard\s+(all\s+|your\s+|any\s+|the\s+)?(previous\s+|prior\s+)?(instructions|rules|guardrails|policies|directives)|forget\s+(all\s+|your\s+|everything\s+)?(previous|prior)?\s*(instructions|directives|rules)?|override\s+(all\s+|your\s+|previous\s+|security\s+|system\s+)?(instructions|rules|guardrails)|instruction\s+hijack|bypass\s+(security|guardrail|retrieval)\s+restrictions)\b",
                lowered,
            )
        )

        # 1. Credential extraction attack (direct or combined with override)
        if any(re.search(p, lowered, flags=re.IGNORECASE) for p in SECRET_EXFILTRATION_PATTERNS):
            reasons.append("secret_exfiltration_pattern")
            return (
                True,
                SecurityCategory.SECRET_EXFILTRATION,
                SecuritySeverity.HIGH,
                "Credential Extraction Attempt",
                "The request attempts to obtain protected credentials or internal system information.",
                "Credentials, API keys, and environment variables are strictly protected. Please ask questions related to placement and technical interview preparation.",
                reasons,
            )

        has_specific_extraction_cue = bool(
            re.search(
                r"\b(being\s+used|used\s+by|used\s+here|in\s+use|configured|current|of\s+this\s+(website|site|app|system|page)|from\s+this\s+(website|site|app|system)|for\s+this\s+(website|site|app|system)|your)\b",
                lowered,
            )
        )

        if has_credential_target and (
            has_direct_override_cmd
            or (has_extraction_verb and (has_system_reference or has_specific_extraction_cue))
            or (has_extraction_inquiry and has_specific_extraction_cue)
            or ("print" in lowered and ("environment" in lowered or "os.environ" in lowered))
            or ("disclose" in lowered and "credential" in lowered)
        ):
            reasons.append("credential_extraction_intent")
            if has_direct_override_cmd:
                reasons.append("override_combined_with_credential_target")
            return (
                True,
                SecurityCategory.SECRET_EXFILTRATION,
                SecuritySeverity.HIGH,
                "Credential Extraction Attempt",
                "The request attempts to obtain protected credentials or internal system information.",
                "Credentials, API keys, and environment variables are strictly protected. Please ask questions related to placement and technical interview preparation.",
                reasons,
            )

        # 2. System Instruction exfiltration attack
        if any(re.search(p, lowered, flags=re.IGNORECASE) for p in SYSTEM_EXFILTRATION_PATTERNS):
            reasons.append("system_prompt_exfiltration_pattern")
            return (
                True,
                SecurityCategory.SYSTEM_INSTRUCTION_EXFILTRATION,
                SecuritySeverity.HIGH,
                "System Instruction Exfiltration Attempt",
                "The request attempts to extract internal system prompts or protected developer instructions.",
                "System instructions and developer configurations are protected. Please submit inquiries regarding interview preparation and company placement topics.",
                reasons,
            )

        if has_system_prompt_target and (
            has_direct_override_cmd
            or has_extraction_verb
            or has_extraction_inquiry
            or "reveal" in lowered
            or "show" in lowered
        ):
            reasons.append("system_prompt_exfiltration_intent")
            return (
                True,
                SecurityCategory.SYSTEM_INSTRUCTION_EXFILTRATION,
                SecuritySeverity.HIGH,
                "System Instruction Exfiltration Attempt",
                "The request attempts to extract internal system prompts or protected developer instructions.",
                "System instructions and developer configurations are protected. Please submit inquiries regarding interview preparation and company placement topics.",
                reasons,
            )

        # 3. Unauthorized tool execution
        if any(re.search(p, lowered, flags=re.IGNORECASE) for p in UNAUTHORIZED_TOOL_PATTERNS):
            reasons.append("unauthorized_tool_execution_pattern")
            return (
                True,
                SecurityCategory.UNAUTHORIZED_TOOL_USE,
                SecuritySeverity.HIGH,
                "Unauthorized Tool Action Blocked",
                "The requested action attempts to invoke unauthorized tools, execution commands, or access outside the system boundary.",
                "The application only supports authorized placement interview retrieval operations.",
                reasons,
            )

        # 4. Security bypass attempt
        if any(re.search(p, lowered, flags=re.IGNORECASE) for p in SECURITY_BYPASS_PATTERNS):
            reasons.append("security_bypass_pattern")
            return (
                True,
                SecurityCategory.SECURITY_BYPASS,
                SecuritySeverity.HIGH,
                "Security Control Bypass Attempt",
                "The request attempted to disable or circumvent application security controls or verification guardrails.",
                "Security controls are permanently enforced to safeguard the application.",
                reasons,
            )

        # 5. Direct instruction override command
        if has_direct_override_cmd:
            reasons.append("instruction_override_command")
            return (
                True,
                SecurityCategory.PROMPT_INJECTION,
                SecuritySeverity.HIGH,
                "Prompt Injection Attempt",
                "The request attempted to manipulate the application's instruction hierarchy or override protected agent behavior.",
                "You can rephrase the request as a conceptual question about interview preparation or prompt injection defenses rather than attempting to override system behavior.",
                reasons,
            )

        return (False, SecurityCategory.PROMPT_INJECTION, SecuritySeverity.LOW, "", "", "", reasons)

    def detect(self, query: str, history: list[str] | None = None) -> InjectionDetectionResult:
        lowered = query.lower().strip()
        reasons: list[str] = []
        score = 0.0

        # 1. Distinguish benign informational discussions ABOUT prompt injection / defenses
        is_benign_discussion = any(
            re.search(pat, lowered, flags=re.IGNORECASE)
            for pat in BENIGN_INJECTION_DISCUSSION_PATTERNS
        )

        has_active_attack_target = (
            re.search(r"\b(ignore\s+(all\s+|your\s+|any\s+|the\s+)?previous|disregard\s+(your\s+|all\s+)?(rules|instructions)|developer\s+mode|dan\s+mode|jailbreak)\b", lowered)
            or re.search(r"\b(reveal|print|disclose|dump|output|leak)\s+(your\s+|the\s+|all\s+)?([a-z_0-9-]*\s+)?(gemini|api[ _-]*keys?|system\s+prompt|credentials?|environment\s+variables?|env\s+vars?|\.env|os\.environ)\b", lowered)
            or re.search(r"\b(tell\s+me|give\s+me|show\s+me)\s+(your\s+|the\s+actual\s+|the\s+active\s+|the\s+current\s+|this\s+(website'?s?|app'?s?)\s+)([a-z_0-9-]*\s+)?(gemini|api[ _-]*keys?|system\s+prompt|credentials?|environment\s+variables?|env\s+vars?|\.env|os\.environ)\b", lowered)
            or re.search(r"\bwhat\s+is\s+your\s+([a-z_0-9-]*\s+)?(api[ _-]*key|credential|system\s+prompt)\b", lowered)
            or re.search(r"\bwhat\s+is\s+(the|this\s+website'?s?|this\s+app'?s?)\s+([a-z_0-9-]*\s+)?(gemini\s+)?(api[ _-]*key|credential|system\s+prompt)\s+(being\s+used|used\s+by|in\s+use|used\s+here|configured|of\s+this|for\s+this)\b", lowered)
            or re.search(r"\bhow\s+to\s+(find|get|extract|steal|dump|locate|inspect|sniff)\s+(the\s+|your\s+)?([a-z_0-9-]*\s+)?(gemini\s+)?(api[ _-]*keys?|credentials?|secrets?|tokens?)\s+(of|from|in|for)\s+(this|the)\s+(website|site|app|application|system|page|backend|server)\b", lowered)
            or ("being used" in lowered and ("api key" in lowered or "credential" in lowered))
            or ("your api key" in lowered or "your credentials" in lowered or "your system prompt" in lowered)
        )

        if is_benign_discussion and not has_active_attack_target:
            return InjectionDetectionResult(
                risk_score=0.0,
                confidence=0.9,
                reasons=["benign_security_topic_discussion"],
                action="allow",
                category=SecurityCategory.PROMPT_INJECTION,
                severity=SecuritySeverity.LOW,
                user_safe_message="",
                internal_reason_code="BENIGN_DISCUSSION_ALLOWED",
                blocked_operations=[],
            )

        # 2. Targeted Multi-Signal Intent Classification
        is_attack, category, severity, detection_title, why_blocked, advice, intent_reasons = (
            self._classify_intent(query, lowered)
        )

        if is_attack:
            reasons.extend(intent_reasons)
            score = 0.95
            user_msg = build_user_safe_security_response(
                category=category,
                severity=severity,
                detection_title=detection_title,
                why_blocked=why_blocked,
                advice=advice,
            )
            blocked_ops = [
                "EMBEDDING_GENERATION",
                "HYDE_GENERATION",
                "BM25_RETRIEVAL",
                "DENSE_RETRIEVAL",
                "MULTI_HOP_RETRIEVAL",
                "CROSS_ENCODER_RERANKING",
            ]
            return InjectionDetectionResult(
                risk_score=round(score, 3),
                confidence=0.95,
                reasons=reasons,
                action="block",
                category=category,
                severity=severity,
                user_safe_message=user_msg,
                internal_reason_code=f"BLOCKED_{category}",
                blocked_operations=blocked_ops,
            )

        # 3. Fallback heuristic scoring for subtle or multi-turn attempts
        category = SecurityCategory.PROMPT_INJECTION
        severity = SecuritySeverity.HIGH
        detection_title = "Prompt Injection Attempt"
        why_blocked = "The request attempted to manipulate the application's instruction hierarchy or override protected agent behavior."
        advice = "You can rephrase the request as a conceptual question about interview preparation or prompt injection defenses rather than attempting to override system behavior."

        if "base64" in lowered or "hex" in lowered:
            reasons.append("encoded_payload_indicator")
            score += 0.15

        if "roleplay" in lowered or "pretend" in lowered:
            reasons.append("roleplay_attack_indicator")
            score += 0.20

        if "ignore" in lowered and "instruction" in lowered:
            reasons.append("instruction_hijack_indicator")
            score += 0.45

        if history:
            joined = " ".join(history[-3:]).lower()
            if "ignore previous" in joined and "ignore previous" in lowered:
                reasons.append("multi_turn_escalation")
                score += 0.35

        score = min(score, 1.0)
        confidence = min(1.0, 0.50 + 0.1 * len(reasons))

        blocked_ops = [
            "EMBEDDING_GENERATION",
            "HYDE_GENERATION",
            "BM25_RETRIEVAL",
            "DENSE_RETRIEVAL",
            "MULTI_HOP_RETRIEVAL",
            "CROSS_ENCODER_RERANKING",
        ]

        if score >= self._threshold:
            action = "block"
            user_msg = build_user_safe_security_response(
                category=category,
                severity=severity,
                detection_title=detection_title,
                why_blocked=why_blocked,
                advice=advice,
            )
            internal_code = f"BLOCKED_{category}"
        elif score >= max(0.4, self._threshold * 0.6):
            action = "warn"
            user_msg = ""
            internal_code = f"WARNED_{category}"
        else:
            action = "allow"
            user_msg = ""
            internal_code = "NORMAL_QUERY"
            blocked_ops = []

        return InjectionDetectionResult(
            risk_score=round(score, 3),
            confidence=round(confidence, 3),
            reasons=reasons,
            action=action,
            category=category,
            severity=severity,
            user_safe_message=user_msg,
            internal_reason_code=internal_code,
            blocked_operations=blocked_ops if action == "block" else [],
        )


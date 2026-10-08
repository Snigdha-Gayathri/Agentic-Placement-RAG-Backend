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
    """Build polished user-facing security response following Section 39.4."""
    return (
        "### 🛡️ Security Validation Notice\n\n"
        "🛡️ **Security Check**\n\n"
        "**Request blocked.**\n\n"
        f"**Detection:**\n{detection_title}\n\n"
        f"**Severity:**\n{severity}\n\n"
        f"**Why it was blocked:**\n{why_blocked}\n\n"
        "**Action taken:**\n"
        "The request was stopped before retrieval and no protected tools, internal instructions, or credentials were exposed.\n\n"
        "**Retrieval:**\n"
        "Not executed\n\n"
        "**Your data:**\n"
        "No protected system information was disclosed.\n\n"
        f"**Guidance:**\n{advice}"
    )


class PromptInjectionDetector:
    """Score query risk for direct and indirect prompt injection patterns with intent separation."""

    def __init__(self, config: SecurityConfig) -> None:
        self._threshold = config.prompt_injection_threshold

    def detect(self, query: str, history: list[str] | None = None) -> InjectionDetectionResult:
        lowered = query.lower().strip()
        reasons: list[str] = []
        score = 0.0

        # 1. Distinguish benign informational discussions ABOUT prompt injection / defenses
        is_benign_discussion = any(
            re.search(pat, lowered, flags=re.IGNORECASE)
            for pat in BENIGN_INJECTION_DISCUSSION_PATTERNS
        )

        has_direct_override = (
            re.search(r"ignore\s+(all\s+)?previous", lowered)
            or re.search(r"reveal\s+(your\s+|the\s+)?(system\s+prompt|prompt|api[_-]?key|secret)", lowered)
            or re.search(r"disregard\s+(your\s+)?(rules|instructions|policies)", lowered)
            or re.search(r"developer\s+mode", lowered)
        )

        if is_benign_discussion and not has_direct_override:
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

        # 2. Identify Category Specific Attacks
        category = SecurityCategory.PROMPT_INJECTION
        severity = SecuritySeverity.HIGH
        detection_title = "Prompt Injection Attempt"
        why_blocked = "The request attempted to manipulate the application's instruction hierarchy or override protected agent behavior."
        advice = "You can rephrase the request as a conceptual question about interview preparation or prompt injection defenses rather than attempting to override system behavior."

        # Check Secret Exfiltration
        if any(re.search(p, lowered, flags=re.IGNORECASE) for p in SECRET_EXFILTRATION_PATTERNS):
            score += 0.85
            reasons.append("secret_exfiltration_pattern")
            category = SecurityCategory.SECRET_EXFILTRATION
            severity = SecuritySeverity.CRITICAL
            detection_title = "Protected Information Request Blocked"
            why_blocked = "The request attempted to obtain credentials, API keys, internal configuration, or other protected information."
            advice = "Credentials and system configuration are protected. Please ask questions related to technical interview preparation."

        # Check System Instruction Exfiltration
        elif any(re.search(p, lowered, flags=re.IGNORECASE) for p in SYSTEM_EXFILTRATION_PATTERNS):
            score += 0.80
            reasons.append("system_prompt_exfiltration_pattern")
            category = SecurityCategory.SYSTEM_INSTRUCTION_EXFILTRATION
            severity = SecuritySeverity.HIGH
            detection_title = "System Instruction Exfiltration Blocked"
            why_blocked = "The request attempted to extract internal system prompts, developer messages, or hidden instructions."
            advice = "System instructions are internal and protected. Please submit inquiries regarding placement and interview topics."

        # Check Unauthorized Tool Use
        elif any(re.search(p, lowered, flags=re.IGNORECASE) for p in UNAUTHORIZED_TOOL_PATTERNS):
            score += 0.80
            reasons.append("unauthorized_tool_execution_pattern")
            category = SecurityCategory.UNAUTHORIZED_TOOL_USE
            severity = SecuritySeverity.HIGH
            detection_title = "Unauthorized Tool Action Blocked"
            why_blocked = "The requested action attempts to invoke unauthorized tools, execution commands, or access outside the system boundary."
            advice = "The application only supports authorized placement interview retrieval operations."

        # Check Security Bypass
        elif any(re.search(p, lowered, flags=re.IGNORECASE) for p in SECURITY_BYPASS_PATTERNS):
            score += 0.85
            reasons.append("security_bypass_pattern")
            category = SecurityCategory.SECURITY_BYPASS
            severity = SecuritySeverity.CRITICAL
            detection_title = "Security Control Bypass Blocked"
            why_blocked = "The request attempted to disable or circumvent application security controls or verification guardrails."
            advice = "Security controls are permanently enforced to safeguard the application."

        # General Prompt Injection / Override Patterns
        else:
            for pattern in PROMPT_INJECTION_PATTERNS:
                if re.search(pattern, lowered, flags=re.IGNORECASE):
                    reasons.append(f"pattern:{pattern}")
                    score += 0.40

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


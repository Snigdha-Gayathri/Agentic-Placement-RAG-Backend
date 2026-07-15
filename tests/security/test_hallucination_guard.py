from security.hallucination_guard import HallucinationGuard


def test_hallucination_guard_flags_unsupported_claims() -> None:
    guard = HallucinationGuard()
    evidence = ["Google asks about system design and distributed systems"]
    answer = "Contact us at fake@example.com and 97% of candidates fail due to hidden policy."
    assert guard.is_hallucinated(answer, evidence, threshold=0.3)


def test_hallucination_guard_allows_supported_answer() -> None:
    guard = HallucinationGuard()
    evidence = ["Google asks system design and distributed systems questions"]
    answer = "Google asks system design and distributed systems questions."
    assert not guard.is_hallucinated(answer, evidence, threshold=0.8)

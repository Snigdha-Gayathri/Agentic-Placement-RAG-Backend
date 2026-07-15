from security.grounding import GroundingVerifier


def test_grounding_fails_without_evidence() -> None:
    verifier = GroundingVerifier()
    assert not verifier.verify("This is an answer", [], 0.2)


def test_grounding_passes_with_evidence() -> None:
    verifier = GroundingVerifier()
    evidence = ["Google asks system design and algorithm questions"]
    answer = "Google asks system design and algorithm questions in interviews."
    assert verifier.verify(answer, evidence, 0.2)

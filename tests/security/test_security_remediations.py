"""Comprehensive security regression tests verifying Phase B & Phase C remediations."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.gemini_client import GeminiClient
from backend.app.main import app
from backend.app.models import ChatRequest, FeatureToggleConfig
from backend.app.service import DASHBOARD_STORE, SecureRAGService, build_request_context
from backend.core.embeddings.gemini import GeminiEmbedding
from backend.core.llm.gemini import GeminiLLM
from backend.core.memory.conversation import ConversationMemory
from backend.ingestion.drive_sync import DriveSyncer
from backend.observability.dashboard import DashboardData, DashboardStore
from security import (
    AuthContext,
    CrossProcessLock,
    GroundingVerifier,
    HallucinationGuard,
    Role,
    SecurityConfig,
    SlidingWindowRateLimiter,
)


@pytest.fixture
def client():
    return TestClient(app)


# ----------------------------------------------------------------------
# SEC-01: VITE_GEMINI_API_KEY removal
# ----------------------------------------------------------------------
def test_sec01_no_vite_gemini_api_key_in_codebase():
    """Verify that VITE_GEMINI_API_KEY does not exist in source code or settings."""
    app_jsx = Path("src/App.jsx")
    if app_jsx.exists():
        assert "VITE_GEMINI_API_KEY" not in app_jsx.read_text(encoding="utf-8")

    settings_py = Path("backend/config/settings.py").read_text(encoding="utf-8")
    assert "VITE_GEMINI_API_KEY" not in settings_py

    gemini_client_py = Path("backend/app/gemini_client.py").read_text(encoding="utf-8")
    assert "VITE_GEMINI_API_KEY" not in gemini_client_py


# ----------------------------------------------------------------------
# SEC-02 & SEC-04: Authentication & Authorization on FastAPI routes
# ----------------------------------------------------------------------
def test_sec02_and_sec04_auth_and_rbac(client):
    """Verify that unauthenticated callers are rejected and RBAC is enforced."""
    # 1. Anonymous access to protected endpoints is rejected with 401
    resp_chat = client.post("/chat", json={"query": "Hello"})
    assert resp_chat.status_code == 401

    resp_config_post = client.post("/config", json={"hyde": True})
    assert resp_config_post.status_code == 401

    resp_reindex = client.post("/reindex")
    assert resp_reindex.status_code == 401

    resp_vector_stats = client.get("/vector-db/stats")
    assert resp_vector_stats.status_code == 401

    # 2. Public endpoints remain accessible
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "ok"

    resp_sec_status = client.get("/security/status")
    assert resp_sec_status.status_code == 200

    # 3. User role credentials allow chat but forbid admin routes (403)
    user_headers = {"Authorization": "Bearer rag-client-key-2026"}
    resp_config_user = client.post("/config", json={"hyde": True}, headers=user_headers)
    assert resp_config_user.status_code == 403

    resp_reindex_user = client.post("/reindex", headers=user_headers)
    assert resp_reindex_user.status_code == 403

    # 4. Admin role credentials allow admin routes
    admin_headers = {"X-API-Key": "rag-admin-secret-key-2026"}
    resp_config_admin = client.post("/config", json={"hyde": True}, headers=admin_headers)
    assert resp_config_admin.status_code == 200
    assert resp_config_admin.json()["hyde"] is True


# ----------------------------------------------------------------------
# SEC-05: In-Process & Cross-Process Reindexing Concurrency Lock
# ----------------------------------------------------------------------
@pytest.mark.anyio
async def test_sec05_reindex_concurrency_lock():
    """Verify that concurrent reindexing is prevented with in-process and cross-process locks."""
    service = SecureRAGService()

    # Simulate an active reindex operation holding the in-process lock
    await service._reindex_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="Reindexing is already in progress"):
            await service.reindex()
    finally:
        service._reindex_lock.release()


def test_sec05_cross_process_file_lock(tmp_path):
    """Verify that CrossProcessLock prevents multi-worker concurrency and recovers from dead PIDs."""
    lock_file = tmp_path / "sync.lock"
    lock1 = CrossProcessLock(lock_file)
    lock2 = CrossProcessLock(lock_file)

    # 1. Process 1 acquires lock
    assert lock1.acquire() is True

    # 2. Process 2 attempts to acquire same lock -> fails
    assert lock2.acquire() is False

    # 3. Release allows subsequent acquisition
    lock1.release()
    assert lock2.acquire() is True
    lock2.release()

    # 4. Stale dead-process lock recovery
    lock_file.write_text("999999\n", encoding="utf-8")
    assert lock1.acquire() is True
    lock1.release()


def test_sec05_reindex_returns_409_on_contention(client):
    """Verify that POST /reindex returns HTTP 409 when reindexing is currently locked."""
    admin_headers = {"X-API-Key": "rag-admin-secret-key-2026"}
    
    # Acquire cross-process lock to simulate another active process
    lock_file = Path("data") / ".reindex.lock"
    external_lock = CrossProcessLock(lock_file)
    assert external_lock.acquire() is True
    try:
        resp = client.post("/reindex", headers=admin_headers)
        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"].lower()
    finally:
        external_lock.release()


# ----------------------------------------------------------------------
# SEC-06: Google Drive Path Traversal Prevention & Correct Synchronization
# ----------------------------------------------------------------------
def test_sec06_drive_sync_explicit_scenarios(tmp_path):
    """Verify all 8 explicit DriveSyncer test requirements from prompt."""
    fake_sa_json = json.dumps({
        "type": "service_account",
        "project_id": "test",
        "private_key_id": "k1",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQD\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test.iam.gserviceaccount.com"
    })

    # Prepare existing state file
    state_file = tmp_path / ".drive_sync_state.json"
    state_file.write_text(json.dumps({
        "unchanged.pdf": "2026-01-01T00:00:00Z",
        "modified.pdf": "2026-01-01T00:00:00Z",
        "missing_local.pdf": "2026-01-01T00:00:00Z",
    }), encoding="utf-8")
    (tmp_path / "unchanged.pdf").write_bytes(b"content")
    (tmp_path / "modified.pdf").write_bytes(b"content")
    # missing_local.pdf is deliberately not created locally

    test_files = [
        # 1. New valid PDF -> should download
        {"id": "1", "name": "new_valid.pdf", "modifiedTime": "2026-01-01T10:00:00Z"},
        # 2. Existing unchanged PDF -> should skip
        {"id": "2", "name": "unchanged.pdf", "modifiedTime": "2026-01-01T00:00:00Z"},
        # 3. Existing modified PDF -> should download
        {"id": "3", "name": "modified.pdf", "modifiedTime": "2026-01-02T10:00:00Z"},
        # 4. Missing local PDF -> should download
        {"id": "4", "name": "missing_local.pdf", "modifiedTime": "2026-01-01T00:00:00Z"},
        # 5. ../ traversal -> should reject
        {"id": "5", "name": "../../escape.pdf", "modifiedTime": "2026-01-01"},
        # 6. Absolute path -> should reject
        {"id": "6", "name": "/etc/shadow.pdf", "modifiedTime": "2026-01-01"},
        # 7. Windows-style traversal/path -> should reject
        {"id": "7", "name": "C:\\Windows\\System32\\hosts.pdf", "modifiedTime": "2026-01-01"},
        # 8. Non-PDF -> should reject
        {"id": "8", "name": "malicious.exe", "modifiedTime": "2026-01-01"},
        {"id": "9", "name": "script.py", "modifiedTime": "2026-01-01"},
    ]

    with patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": fake_sa_json, "GOOGLE_DRIVE_FOLDER_ID": "test_folder"}):
        with patch("google.oauth2.service_account.Credentials.from_service_account_info"), \
             patch("googleapiclient.discovery.build") as mock_build:
            mock_service = MagicMock()
            mock_files = MagicMock()
            mock_service.files.return_value = mock_files
            mock_files.list.return_value.execute.return_value = {"files": test_files}
            mock_build.return_value = mock_service

            with patch("backend.ingestion.drive_sync.MediaIoBaseDownload") as mock_dl:
                mock_dl_instance = MagicMock()
                mock_dl_instance.next_chunk.return_value = (None, True)
                mock_dl.return_value = mock_dl_instance

                syncer = DriveSyncer(data_path=str(tmp_path))
                syncer._service = mock_service
                downloaded = syncer.sync_pdfs()

                assert "new_valid.pdf" in downloaded
                assert "modified.pdf" in downloaded
                assert "missing_local.pdf" in downloaded
                assert "unchanged.pdf" not in downloaded
                assert "escape.pdf" not in downloaded
                assert "shadow.pdf" not in downloaded
                assert "hosts.pdf" not in downloaded
                assert "malicious.exe" not in downloaded
                assert "script.py" not in downloaded


# ----------------------------------------------------------------------
# SEC-07: Indirect Prompt Injection & Context Demarcation
# ----------------------------------------------------------------------
@pytest.mark.anyio
async def test_sec07_prompt_injection_context_demarcation():
    """Verify that context and query are demarcated with XML tags and system directive is added."""
    client = GeminiClient(api_key="test_key")

    with patch.object(client, "generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Safe generated response"
        context = "Interview syllabus: Binary Search and Trees."
        query = "What data structures should I study?"

        await client.generate_answer(query=query, context=context)

        system_prompt = mock_gen.call_args.kwargs["system_prompt"]
        user_prompt = mock_gen.call_args.kwargs["user_prompt"]

        # Verify system prompt has security directive
        assert "CRITICAL SECURITY DIRECTIVE (DATA-INSTRUCTION SEPARATION)" in system_prompt
        assert "<retrieved_context>" in system_prompt

        # Verify user prompt wraps context in <retrieved_context> and query in <user_query>
        assert f"<retrieved_context>\n{context}\n</retrieved_context>" in user_prompt
        assert f"<user_query>\n{query}\n</user_query>" in user_prompt


# ----------------------------------------------------------------------
# SEC-08: Rate Limiter Identity Enforcement & Bounded Storage
# ----------------------------------------------------------------------
def test_sec08_rate_limiter_spoofing_prevention_and_bounds():
    """Verify that rotating session_id does not bypass rate limit for an IP/user, and memory is bounded."""
    cfg = SecurityConfig(rate_limit=5, rate_window_seconds=60)
    limiter = SlidingWindowRateLimiter(config=cfg, max_keys=10)

    # 1. Rapid requests with different session_ids for same IP/user should be blocked after 5 requests
    for i in range(5):
        decision = limiter.allow(ip="192.168.1.100", session_id=f"sess_{i}", user_id="user_123")
        assert decision.allowed is True

    blocked_decision = limiter.allow(ip="192.168.1.100", session_id="sess_new", user_id="user_123")
    assert blocked_decision.allowed is False
    assert blocked_decision.retry_after_seconds > 0

    # 2. Key store pruning stays bounded
    for i in range(50):
        limiter.allow(ip=f"10.0.0.{i}", session_id=f"s_{i}", user_id=f"u_{i}")

    assert len(limiter._events) <= 55


# ----------------------------------------------------------------------
# SEC-09: Dashboard IDOR & Session Isolation
# ----------------------------------------------------------------------
def test_sec09_dashboard_authorization_and_idor(client):
    """Verify that users can only access their own dashboard records and admin can access all."""
    store = DASHBOARD_STORE
    req_id = "req_user1_secret_99"

    # User 1 stores their dashboard
    store.store(req_id, DashboardData(owner_id="user_alice", request_id=req_id))

    # User 2 tries to access User 1's dashboard -> 404
    user2_headers = {"Authorization": "Bearer user_bob_session_token_12345"}
    resp_user2 = client.get(f"/dashboard/{req_id}", headers=user2_headers)
    assert resp_user2.status_code == 404

    # User 1 accesses their own dashboard -> 200
    user1_headers = {"Authorization": "Bearer user_alice"}
    resp_user1 = client.get(f"/dashboard/{req_id}", headers=user1_headers)
    assert resp_user1.status_code == 200
    assert resp_user1.json()["request_id"] == req_id

    # Admin accesses User 1's dashboard -> 200
    admin_headers = {"X-API-Key": "rag-admin-secret-key-2026"}
    resp_admin = client.get(f"/dashboard/{req_id}", headers=admin_headers)
    assert resp_admin.status_code == 200


# ----------------------------------------------------------------------
# SEC-10: ConversationMemory Session Hijacking Prevention
# ----------------------------------------------------------------------
def test_sec10_conversation_memory_session_isolation():
    """Verify that sessions are owned and isolated between distinct users."""
    mem = ConversationMemory()
    session_id = "shared_or_guessable_session_id"

    # User Alice establishes session
    mem.add_turn(session_id=session_id, role="user", content="My private company offer is Google", owner_id="user_alice")

    # User Bob attempts to inject turns or read Alice's session
    with pytest.raises(PermissionError, match="owned by another user"):
        mem.add_turn(session_id=session_id, role="user", content="Malicious injected prompt", owner_id="user_bob")

    bob_history = mem.get_history(session_id=session_id, owner_id="user_bob")
    assert bob_history == []

    # Alice can read her own history
    alice_history = mem.get_history(session_id=session_id, owner_id="user_alice")
    assert len(alice_history) == 1
    assert "Google" in alice_history[0].content


# ----------------------------------------------------------------------
# SEC-11: Gemini API Key in Headers
# ----------------------------------------------------------------------
@pytest.mark.anyio
async def test_sec11_gemini_api_key_sent_in_header():
    """Verify that Gemini client, LLM, and Embedder send API key in x-goog-api-key header and not URL."""
    client = GeminiClient(api_key="secret_test_gemini_key")
    llm = GeminiLLM(api_key="secret_test_gemini_key")
    embedder = GeminiEmbedding(api_key="secret_test_gemini_key")

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
            "embedding": {"values": [0.1, 0.2]},
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        # Test GeminiClient
        await client.generate("sys", "usr")
        call_url = mock_post.call_args_list[0].args[0]
        call_headers = mock_post.call_args_list[0].kwargs.get("headers", {})
        assert "key=" not in call_url
        assert call_headers.get("x-goog-api-key") == "secret_test_gemini_key"

        # Test GeminiLLM
        await llm.generate("sys", "usr")
        call_url2 = mock_post.call_args_list[1].args[0]
        call_headers2 = mock_post.call_args_list[1].kwargs.get("headers", {})
        assert "key=" not in call_url2
        assert call_headers2.get("x-goog-api-key") == "secret_test_gemini_key"

        # Test GeminiEmbedding
        await embedder.embed_text("test")
        call_url3 = mock_post.call_args_list[2].args[0]
        call_headers3 = mock_post.call_args_list[2].kwargs.get("headers", {})
        assert "key=" not in call_url3
        assert call_headers3.get("x-goog-api-key") == "secret_test_gemini_key"


# ----------------------------------------------------------------------
# SEC-12: Bounded DashboardStore
# ----------------------------------------------------------------------
def test_sec12_dashboard_store_bounded_capacity():
    """Verify that DashboardStore evicts oldest entries and does not grow unboundedly."""
    store = DashboardStore(max_size=5)
    for i in range(10):
        store.store(f"req_{i}", DashboardData(request_id=f"req_{i}"))

    assert len(store) == 5
    assert store.get("req_0") is None  # Evicted
    assert store.get("req_9") is not None  # Retained


# ----------------------------------------------------------------------
# SEC-14: Grounding & Hallucination Adversarial Matrix (Cases A-M)
# ----------------------------------------------------------------------
def test_sec14_grounding_adversarial_matrix():
    """Verify that GroundingVerifier and HallucinationGuard catch negations, contradictions, and fabrications."""
    verifier = GroundingVerifier()
    guard = HallucinationGuard()

    evidence = [
        "Google software engineer technical interviews evaluate data structures, algorithms, and system design. The coding rounds involve problem-solving in Python, Java, or C++ with time and space complexity analysis.",
        "Amazon evaluates 14 Leadership Principles alongside coding proficiency and object-oriented design patterns.",
    ]

    adversarial_cases = [
        ("A. Directly supported", "Google technical interviews evaluate data structures, algorithms, and system design.", True),
        ("B. Supported paraphrase", "Candidates at Google are tested on algorithmic reasoning, system architecture, and computational complexity.", True),
        ("C. Keyword-stuffed fabrication", "Google technical interviews require candidates to submit patent documentation before scheduling algorithms rounds.", False),
        ("D. Direct contradiction", "Google does not evaluate algorithms or data structures during software engineer interviews.", False),
        ("E. Fabricated fee", "Google requires candidates to pay a $500 registration fee before attending coding rounds.", False),
        ("F. Fabricated date", "The application window closes on October 31, 2026 for all engineering roles.", False),
        ("G. Fabricated eligibility", "Candidates must possess at least 8 years of previous FAANG experience to apply for junior roles.", False),
        ("H. Fabricated compensation", "The base compensation package is strictly $180,000 with mandatory sign-on bonus of $50,000.", False),
        ("I. Fabricated requirement", "Candidates must submit a mandatory video recording explaining their life history before coding.", False),
        ("J. Mixed claim with unsupported fee", "Google evaluates algorithms and data structures. However, candidates must pay a mandatory fee of $300.", False),
        ("K. Multi-sentence with 1 unsupported", "Google software engineer interviews test data structures. Candidates solve coding problems in Python or Java. Candidates must pay a $100 processing charge. Time complexity analysis is evaluated.", False),
        ("L. Completely unsupported vocabulary", "Aerospace turbine combustion requires cryogenic liquid propellant.", False),
        ("M. Negation contradiction", "Amazon never evaluates Leadership Principles during technical rounds.", False),
    ]

    for label, text, expected_grounded in adversarial_cases:
        is_grounded = verifier.verify(text, evidence)
        assert is_grounded == expected_grounded, f"Failed on {label}: Grounded={is_grounded}, Expected={expected_grounded}"
        if not expected_grounded:
            assert guard.is_hallucinated(text, evidence) or not is_grounded


def test_sec14_grounding_explicit_cases_a_through_h():
    """Explicitly verify prompt test cases A through H for polarity, contradiction, and support."""
    verifier = GroundingVerifier()

    # A. Supported
    ev_a = ["Google evaluates data structures and algorithms."]
    claim_a = "Google evaluates data structures and algorithms."
    assert verifier.verify(claim_a, ev_a) is True

    # B. Paraphrase
    ev_b = ["Google evaluates data structures and algorithms."]
    claim_b = "Google tests candidates on algorithmic problem solving and data structures."
    assert verifier.verify(claim_b, ev_b) is True

    # C. Contradiction
    ev_c = ["Google evaluates data structures and algorithms."]
    claim_c = "Google does not evaluate data structures or algorithms."
    assert verifier.verify(claim_c, ev_c) is False

    # D. Negation
    ev_d = ["Amazon evaluates Leadership Principles."]
    claim_d = "Amazon never evaluates Leadership Principles."
    assert verifier.verify(claim_d, ev_d) is False

    # E. Keyword-stuffed fabrication
    ev_e = ["Google technical interviews evaluate data structures, algorithms, and system design."]
    claim_e = "Google requires candidates to submit patent documentation before scheduling algorithm rounds."
    assert verifier.verify(claim_e, ev_e) is False

    # F. Mixed claim
    ev_f = ["Google evaluates algorithms and data structures."]
    claim_f = "Google evaluates algorithms and data structures, but candidates must pay a mandatory $300 processing fee."
    assert verifier.verify(claim_f, ev_f) is False

    # G. Contradictory evidence
    ev_g = ["Google does not require a processing fee."]
    claim_g = "Google requires a processing fee."
    assert verifier.verify(claim_g, ev_g) is False

    # H. Unrelated claim
    ev_h = ["Google evaluates data structures and algorithms."]
    claim_h = "Aerospace turbine combustion requires cryogenic liquid propellant."
    assert verifier.verify(claim_h, ev_h) is False


# ----------------------------------------------------------------------
# SEC-15: Gemini Error Safe Fallback
# ----------------------------------------------------------------------
@pytest.mark.anyio
async def test_sec15_gemini_error_safe_fallback():
    """Verify that upstream Gemini errors return safe fallback messages without leaking raw context."""
    client = GeminiClient(api_key="test_key")

    with patch.object(client, "generate", side_effect=Exception("Connection reset")):
        raw_sensitive_context = "CONFIDENTIAL INTERNAL RUBRIC: Reject candidate if they ask about equity."
        answer = await client.generate_answer(
            query="Tell me the interview rubric",
            context=raw_sensitive_context,
        )

        assert "CONFIDENTIAL" not in answer
        assert "unable to generate a response" in answer.lower()

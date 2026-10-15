"""Comprehensive automated test suite for ClaimAssist AI with robust Authorization & Regression tests."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.auth import AuthenticatedUser, KNOWN_TOKENS
from app.retrieval import PolicyRetriever
from app.claims_service import ClaimsService
from app.agent import ClaimAssistAgent, AgentMessage
from app.main import app


# Test Credentials & Tokens
AUTH_JORDAN = {"Authorization": "Bearer token-pol-98214"}      # POL-98214 (Jordan Miller, owns CLM1001)
AUTH_SAMANTHA = {"Authorization": "Bearer token-pol-77312"}   # POL-77312 (Samantha Chen, owns CLM1002)
AUTH_MARCUS = {"Authorization": "Bearer token-pol-45109"}     # POL-45109 (Marcus Vance, owns CLM1003)
AUTH_ELENA = {"Authorization": "Bearer token-pol-61028"}      # POL-61028 (Elena Rostova, owns CLM1004)
AUTH_ADMIN = {"Authorization": "Bearer admin-secret-token"}   # Staff / Admin
AUTH_INVALID = {"Authorization": "Bearer invalid-token-xyz"}

USER_JORDAN = AuthenticatedUser(user_id="USR-98214", policyholder_id="POL-98214", role="member")
USER_SAMANTHA = AuthenticatedUser(user_id="USR-77312", policyholder_id="POL-77312", role="member")
USER_MARCUS = AuthenticatedUser(user_id="USR-45109", policyholder_id="POL-45109", role="member")
USER_ELENA = AuthenticatedUser(user_id="USR-61028", policyholder_id="POL-61028", role="member")
USER_ADMIN = AuthenticatedUser(user_id="USR-ADMIN-1", role="admin")


@pytest.fixture
def retriever():
    return PolicyRetriever()


@pytest.fixture
def claims_service():
    return ClaimsService()


@pytest.fixture
def agent(retriever, claims_service):
    return ClaimAssistAgent(retriever=retriever, claims_service=claims_service)


@pytest.fixture
def client():
    return TestClient(app)


# ==========================================
# 1. Config Tests
# ==========================================
def test_config_defaults():
    assert settings.APP_NAME == "ClaimAssist AI"
    assert settings.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert "claude" in settings.OPENROUTER_MODEL.lower() or "sonnet" in settings.OPENROUTER_MODEL.lower()


# ==========================================
# 2. Retrieval Tests
# ==========================================
def test_policy_retrieval_preexisting(retriever):
    results = retriever.search("pre-existing conditions waiting period", top_k=3)
    assert len(results) > 0
    top_chunk = results[0].chunk
    assert "3.2" in top_chunk.section_id or "pre-existing" in top_chunk.content.lower()
    assert "[Section" in top_chunk.citation


def test_policy_retrieval_cosmetic_exclusion(retriever):
    results = retriever.search("is cosmetic rhinoplasty covered?", top_k=3)
    assert len(results) > 0
    top_chunk = results[0].chunk
    assert "4.1" in top_chunk.section_id or "cosmetic" in top_chunk.content.lower()


def test_policy_retrieval_inpatient(retriever):
    results = retriever.search("inpatient hospitalization ICU surgery room boarding", top_k=3)
    assert len(results) > 0
    found_inpatient = any("2.1" in r.chunk.section_id for r in results)
    assert found_inpatient


# ==========================================
# 3. Claims Service & Authorization Tests
# ==========================================
def test_claims_lookup(claims_service):
    claim = claims_service.get_claim("CLM1001")
    assert claim is not None
    assert claim.policyholder_name == "Jordan Miller"
    assert claim.status == "Approved"
    assert claim.approved_amount == 1350.00


def test_claims_search(claims_service):
    results = claims_service.search_claims("Marcus")
    assert len(results) == 1
    assert results[0].claim_id == "CLM1003"
    assert results[0].status == "Additional Documents Required"


def test_claims_list_for_user_scoping(claims_service):
    """Verify backend queries return ONLY claims belonging to the authorized user."""
    # Jordan Miller only sees POL-98214 claims
    jordan_claims = claims_service.list_claims_for_user(policyholder_id="POL-98214", is_admin=False)
    assert len(jordan_claims) >= 1
    assert all(c.policyholder_id == "POL-98214" for c in jordan_claims)

    # Samantha Chen only sees POL-77312 claims
    samantha_claims = claims_service.list_claims_for_user(policyholder_id="POL-77312", is_admin=False)
    assert len(samantha_claims) >= 1
    assert all(c.policyholder_id == "POL-77312" for c in samantha_claims)

    # Admin sees all claims
    admin_claims = claims_service.list_claims_for_user(policyholder_id=None, is_admin=True)
    assert len(admin_claims) == 6

    # Unauthenticated user receives empty list
    guest_claims = claims_service.list_claims_for_user(policyholder_id=None, is_admin=False)
    assert guest_claims == []


def test_reimbursement_calculation(claims_service):
    res = claims_service.calculate_reimbursement(
        incurred_charges=3000.0,
        deductible_remaining=500.0,
        non_covered_charges=0.0,
        in_network=True
    )
    assert res["estimated_reimbursement"] == 2000.00
    assert res["patient_estimated_responsibility"] == 1000.00


def test_escalation_ticket(claims_service):
    ticket = claims_service.escalate_claim("CLM1004", reason="Disputing denial")
    assert ticket.ticket_id.startswith("ESC-")
    assert ticket.claim_id == "CLM1004"
    assert ticket.status == "Escalated to Specialist"


def test_hipaa_claim_access_authorization(claims_service):
    """Verify HIPAA authorization logic prevents unauthorized PII access."""
    # CLM1001 belongs to Jordan Miller (POL-98214)
    is_auth, msg = claims_service.verify_claim_access("CLM1001", requester_id="POL-98214")
    assert is_auth is True

    # Staff / Admin is authorized
    is_auth_admin, _ = claims_service.verify_claim_access("CLM1001", requester_id=None, is_admin=True)
    assert is_auth_admin is True

    # Cross-member IDOR attempt is blocked
    is_auth_stranger, msg = claims_service.verify_claim_access("CLM1001", requester_id="POL-77312")
    assert is_auth_stranger is False
    assert "Access Denied" in msg

    # Using claim ID as identity spoofing is blocked
    is_auth_claim_spoof, _ = claims_service.verify_claim_access("CLM1001", requester_id="CLM1001")
    assert is_auth_claim_spoof is False


# ==========================================
# 4. Agent Tests & Tool Authorization Tests
# ==========================================
def test_agent_claim_inquiry(agent):
    # Samantha Chen checking her own claim CLM1002
    response = agent.chat("What is the status of claim CLM1002?", provider_override="mock", current_user=USER_SAMANTHA)
    assert "CLM1002" in response.reply
    assert "Samantha Chen" in response.reply
    assert "Under Review" in response.reply
    assert len(response.citations) > 0


def test_agent_claim_inquiry_cross_member_blocked(agent):
    # Samantha Chen attempting to query Jordan Miller's claim CLM1001
    response = agent.chat("What is the status of claim CLM1001?", provider_override="mock", current_user=USER_SAMANTHA)
    assert "HIPAA Access Denied" in response.reply
    assert "CLM1001" in response.reply


def test_agent_claim_inquiry_unauthenticated_blocked(agent):
    # Unauthenticated inquiry for a specific claim
    response = agent.chat("What is the status of claim CLM1001?", provider_override="mock", current_user=None)
    assert "Authentication Required" in response.reply or "Authorization error" in response.reply


def test_agent_tools_strict_authorization(agent):
    """Verify agent tools enforce authentication and member authorization."""
    # 1. Unauthenticated call to lookup_claim
    res_unauth, _ = agent._execute_tool("lookup_claim", {"claim_id": "CLM1001"}, current_user=None)
    assert "error" in res_unauth
    assert "Authentication required" in res_unauth["error"]

    # 2. Cross-member unauthorized lookup
    res_forbidden, _ = agent._execute_tool("lookup_claim", {"claim_id": "CLM1001"}, current_user=USER_SAMANTHA)
    assert "error" in res_forbidden
    assert "Access denied" in res_forbidden["error"]

    # 3. Authorized member lookup
    res_ok, cites = agent._execute_tool("lookup_claim", {"claim_id": "CLM1001"}, current_user=USER_JORDAN)
    assert "claim_id" in res_ok
    assert res_ok["claim_id"] == "CLM1001"
    assert len(cites) > 0

    # 4. Admin lookup
    res_admin, _ = agent._execute_tool("lookup_claim", {"claim_id": "CLM1001"}, current_user=USER_ADMIN)
    assert res_admin["claim_id"] == "CLM1001"

    # 5. Cross-member escalate blocked
    res_esc_block, _ = agent._execute_tool("escalate_to_human_specialist", {"claim_id": "CLM1001", "reason": "Test"}, current_user=USER_SAMANTHA)
    assert "error" in res_esc_block

    # 6. Cross-member audit blocked
    res_audit_block, _ = agent._execute_tool("detect_claim_anomalies", {"claim_id": "CLM1001"}, current_user=USER_SAMANTHA)
    assert "error" in res_audit_block

    # 7. Cross-member appeal blocked
    res_appeal_block, _ = agent._execute_tool("generate_appeal_letter", {"claim_id": "CLM1001"}, current_user=USER_SAMANTHA)
    assert "error" in res_appeal_block


def test_agent_rejection_and_escalation(agent):
    # Elena checking her own rejected claim CLM1004
    response = agent.chat("Check claim CLM1004 and tell me why it was rejected", provider_override="mock", current_user=USER_ELENA)
    assert "CLM1004" in response.reply
    assert "Rejected" in response.reply
    assert "ESC-" in response.reply or any("8.1" in c or "8.2" in c for c in response.citations)


def test_agent_policy_question_citations(agent):
    response = agent.chat("What is the waiting period for maternity and childbirth?", provider_override="mock")
    assert "10-month" in response.reply.lower() or "maternity" in response.reply.lower()
    assert any("3.3" in c for c in response.citations)


def test_mock_agent_multi_turn_context(agent):
    """Verify mock agent extracts active claim from history and executes follow-up actions with auth."""
    history = [
        AgentMessage(role="user", content="What is the status of claim CLM1004?"),
        AgentMessage(role="assistant", content="Claim CLM1004 was rejected for Cosmetic Rhinoplasty.")
    ]
    resp = agent.chat(
        "Can you generate an appeal letter for it?",
        conversation_history=history,
        provider_override="mock",
        current_user=USER_ELENA
    )
    assert "CLM1004" in resp.reply
    assert "FORMAL NOTICE OF CLAIM DISPUTE & APPEAL" in resp.reply
    assert any(t.get("tool") == "generate_appeal_letter" for t in resp.tool_calls)


# ==========================================
# 5. FastAPI Endpoints & RBAC Regression Tests
# ==========================================
def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["policy_chunks_indexed"] > 0
    assert data["claims_loaded"] == 6


# --- Authentication Enforcement (401) ---

def test_api_unauthenticated_requests_return_401(client):
    """Verify all protected endpoints return 401 when no token is supplied."""
    assert client.get("/api/claims").status_code == 401
    assert client.get("/api/claims/CLM1001").status_code == 401
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    assert client.post("/api/claims/CLM1001/escalate", json={"reason": "test"}).status_code == 401
    assert client.get("/api/claims/CLM1001/anomaly").status_code == 401
    assert client.post("/api/claims/CLM1001/appeal-letter").status_code == 401


def test_api_invalid_token_returns_401(client):
    """Verify invalid or expired Bearer tokens return 401."""
    res = client.get("/api/claims", headers=AUTH_INVALID)
    assert res.status_code == 401
    assert "Invalid authentication credentials" in res.json()["detail"]


# --- Claims List Authorization & Scoping ---

def test_api_claims_list_scoping(client):
    """Verify claims-list endpoint returns only the claims the authenticated user is permitted to see."""
    # Jordan Miller (POL-98214) sees only their own claims
    res_jordan = client.get("/api/claims", headers=AUTH_JORDAN)
    assert res_jordan.status_code == 200
    claims_jordan = res_jordan.json()
    assert len(claims_jordan) >= 1
    assert all(c["policyholder_id"] == "POL-98214" for c in claims_jordan)
    # Does not see Samantha Chen's claim CLM1002
    assert not any(c["claim_id"] == "CLM1002" for c in claims_jordan)

    # Samantha Chen (POL-77312) sees only their own claims
    res_samantha = client.get("/api/claims", headers=AUTH_SAMANTHA)
    assert res_samantha.status_code == 200
    claims_samantha = res_samantha.json()
    assert len(claims_samantha) >= 1
    assert all(c["policyholder_id"] == "POL-77312" for c in claims_samantha)
    # Does not see Jordan Miller's claim CLM1001
    assert not any(c["claim_id"] == "CLM1001" for c in claims_samantha)

    # Admin sees all claims
    res_admin = client.get("/api/claims", headers=AUTH_ADMIN)
    assert res_admin.status_code == 200
    claims_admin = res_admin.json()
    assert len(claims_admin) == 6


# --- Claim Detail Endpoint & IDOR Protection (403, 404, 200) ---

def test_api_claim_detail_authorized_owner(client):
    """Verify authorized member can view their own claim."""
    res = client.get("/api/claims/CLM1001", headers=AUTH_JORDAN)
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == "CLM1001"
    assert data["policyholder_name"] == "Jordan Miller"
    assert data["status"] == "Approved"


def test_api_claim_detail_authorized_admin(client):
    """Verify admin / claims specialist can view any claim."""
    res = client.get("/api/claims/CLM1001", headers=AUTH_ADMIN)
    assert res.status_code == 200
    assert res.json()["claim_id"] == "CLM1001"


def test_api_claim_detail_cross_member_idor_returns_403(client):
    """Verify authenticated user is forbidden (403) from accessing another member's claim."""
    # Samantha Chen trying to view Jordan Miller's claim CLM1001
    res = client.get("/api/claims/CLM1001", headers=AUTH_SAMANTHA)
    assert res.status_code == 403
    assert "Access forbidden" in res.json()["detail"]


def test_api_claim_detail_query_param_spoof_fails(client):
    """Verify query parameter ?requester_id cannot spoof identity to bypass authorization."""
    # Samantha supplies Jordan's policyholder ID in query param; must still be 403 based on Bearer token
    res = client.get("/api/claims/CLM1001?requester_id=POL-98214", headers=AUTH_SAMANTHA)
    assert res.status_code == 403


def test_api_claim_detail_not_found_returns_404(client):
    """Verify non-existent claim returns 404."""
    res = client.get("/api/claims/CLM9999", headers=AUTH_JORDAN)
    assert res.status_code == 404


# --- Claim Actions Endpoints Authorization ---

def test_api_escalate_claim_authorization(client):
    # Samantha trying to escalate Jordan's claim -> 403
    res_fail = client.post(
        "/api/claims/CLM1001/escalate",
        headers=AUTH_SAMANTHA,
        json={"reason": "Disputing bill reduction"}
    )
    assert res_fail.status_code == 403

    # Jordan escalating own claim -> 200
    res_ok = client.post(
        "/api/claims/CLM1001/escalate",
        headers=AUTH_JORDAN,
        json={"reason": "Customer requests live review"}
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["claim_id"] == "CLM1001"
    assert res_ok.json()["status"] == "Escalated to Specialist"


def test_api_anomaly_audit_authorization(client):
    # Samantha trying to audit Jordan's claim -> 403
    res_fail = client.get("/api/claims/CLM1001/anomaly", headers=AUTH_SAMANTHA)
    assert res_fail.status_code == 403

    # Jordan auditing own claim -> 200
    res_ok = client.get("/api/claims/CLM1001/anomaly", headers=AUTH_JORDAN)
    assert res_ok.status_code == 200
    assert res_ok.json()["claim_id"] == "CLM1001"


def test_api_appeal_letter_authorization(client):
    # Samantha trying to generate appeal for Jordan's claim -> 403
    res_fail = client.post("/api/claims/CLM1001/appeal-letter", headers=AUTH_SAMANTHA)
    assert res_fail.status_code == 403

    # Jordan generating appeal for own claim -> 200
    res_ok = client.post("/api/claims/CLM1001/appeal-letter", headers=AUTH_JORDAN)
    assert res_ok.status_code == 200
    assert res_ok.json()["claim_id"] == "CLM1001"


def test_api_chat_authenticated(client):
    payload = {
        "message": "What is the waiting period for pre-existing conditions?",
        "provider": "mock"
    }
    res = client.post("/api/chat", json=payload, headers=AUTH_JORDAN)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["citations"]) > 0


def test_api_reimbursement(client):
    payload = {
        "incurred_charges": 2000.0,
        "deductible_remaining": 0.0,
        "non_covered_charges": 0.0,
        "in_network": True
    }
    res = client.post("/api/reimbursement/calculate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["estimated_reimbursement"] == 1600.00


# ==========================================
# 6. Advanced Algorithm Tests
# ==========================================
def test_anomaly_detection():
    from app.algorithms.fraud_detector import ClaimAnomalyDetector
    report = ClaimAnomalyDetector.evaluate_claim(
        claim_id="CLM1004",
        claim_type="Cosmetic Rhinoplasty",
        claim_amount=4500.0,
        submission_date="2026-09-15"
    )
    assert report.claim_id == "CLM1004"
    assert report.risk_level in ["MEDIUM", "HIGH"]
    assert any("cosmetic" in f.lower() for f in report.flags)


def test_benefit_accumulator():
    from app.algorithms.accumulator import BenefitAccumulator
    acc = BenefitAccumulator(persist=False)
    # First claim: $800 bill, deductible is $1000
    res1 = acc.adjudicate_and_accumulate("POL-TEST-CLEAN", 800.0, in_network=True)
    assert res1["deductible_applied"] == 800.0
    assert res1["deductible_remaining_after"] == 200.0
    assert res1["plan_payout"] == 0.0

    # Second claim: $1000 bill. Remaining deductible is $200.
    res2 = acc.adjudicate_and_accumulate("POL-TEST-CLEAN", 1000.0, in_network=True)
    assert res2["deductible_applied"] == 200.0
    assert res2["deductible_remaining_after"] == 0.0
    assert res2["plan_payout"] == 640.0


def test_faithfulness_guardrail():
    from app.algorithms.faithfulness import FaithfulnessGuardrail
    good_reply = "Under [Section 3.2] Pre-Existing Conditions, there is a 12-month waiting period."
    context = "Section 3.2 Pre-Existing Conditions: 12 months waiting period."
    audit = FaithfulnessGuardrail.evaluate(good_reply, context)
    assert audit.is_faithful is True
    assert "[Section 3.2]" in audit.valid_citations

    # Test hallucinated section detection
    hallucinated_reply = "Per [Section 99.9], everything is 100% free with no limits."
    audit_bad = FaithfulnessGuardrail.evaluate(hallucinated_reply, context)
    assert "[Section 99.9]" in audit_bad.invalid_citations
    assert audit_bad.warning_flag is True


def test_appeal_letter_generation(claims_service):
    letter = claims_service.generate_appeal_letter("CLM1004", "Cosmetic reconstruction after nasal trauma")
    assert letter["claim_id"] == "CLM1004"
    assert "FORMAL NOTICE OF CLAIM DISPUTE" in letter["letter_content"]
    assert "Section 8.1" in letter["letter_content"]


def test_pre_claim_adjudication(claims_service):
    res_cosmetic = claims_service.adjudicate_pre_claim("Cosmetic Rhinoplasty", 4000.0)
    assert res_cosmetic["decision"] == "Excluded"
    assert res_cosmetic["is_eligible"] is False

    res_er = claims_service.adjudicate_pre_claim("Emergency Appendix Surgery", 3000.0)
    assert res_er["decision"] == "Covered"
    assert res_er["is_eligible"] is True


def test_sqlite_persistence_and_sync(tmp_path):
    from app.db import Database
    from app.algorithms.accumulator import BenefitAccumulator

    test_db = Database(db_path=tmp_path / "test.db")
    acc1 = BenefitAccumulator(db_instance=test_db)
    acc1.adjudicate_and_accumulate("POL-PERSIST", 700.0, in_network=True)

    acc2 = BenefitAccumulator(db_instance=test_db)
    state = acc2.get_or_create_state("POL-PERSIST")
    assert state.deductible_satisfied == 700.0
    assert state.deductible_remaining == 300.0


def test_duplicate_claim_anomaly_detection():
    from app.algorithms.fraud_detector import ClaimAnomalyDetector
    prior_claims = [
        {"claim_id": "CLM9990", "claim_type": "Emergency Room", "claim_amount": 1200.0, "submission_date": "2026-09-01"}
    ]
    report = ClaimAnomalyDetector.evaluate_claim(
        claim_id="CLM9991",
        claim_type="Emergency Room",
        claim_amount=1200.0,
        submission_date="2026-09-10",
        prior_claims=prior_claims
    )
    assert report.is_potential_duplicate is True
    assert "CLM9990" in report.duplicate_claim_ids
    assert any("Duplicate Claim" in f for f in report.flags)
    assert report.risk_score >= 0.40


def test_numerical_entailment_guardrail():
    from app.algorithms.faithfulness import FaithfulnessGuardrail
    context = "Under Section 3.1, the general waiting period is 30 days."
    hallucinated_reply = "According to Section 3.1, your waiting period is 180 days."
    audit = FaithfulnessGuardrail.evaluate(hallucinated_reply, context)
    assert audit.is_faithful is False
    assert any("180" in num for num in audit.unsupported_numeric_facts)

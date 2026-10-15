"""Comprehensive automated test suite for ClaimAssist AI."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.retrieval import PolicyRetriever
from app.claims_service import ClaimsService
from app.agent import ClaimAssistAgent, AgentMessage
from app.main import app


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


# 1. Config Tests
def test_config_defaults():
    assert settings.APP_NAME == "ClaimAssist AI"
    assert settings.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert "claude" in settings.OPENROUTER_MODEL.lower() or "sonnet" in settings.OPENROUTER_MODEL.lower()


# 2. Retrieval Tests
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


# 3. Claims Service Tests
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


def test_reimbursement_calculation(claims_service):
    # $3000 charge, $500 deductible remaining, in-network (80%)
    # Eligible = 3000, after deductible = 2500, 80% = 2000
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


# 4. Agent Tests (Mock Mode)
def test_agent_claim_inquiry(agent):
    response = agent.chat("What is the status of claim CLM1002?", provider_override="mock")
    assert "CLM1002" in response.reply
    assert "Samantha Chen" in response.reply
    assert "Under Review" in response.reply
    assert len(response.citations) > 0


def test_agent_rejection_and_escalation(agent):
    response = agent.chat("Check claim CLM1004 and tell me why it was rejected", provider_override="mock")
    assert "CLM1004" in response.reply
    assert "Rejected" in response.reply
    assert "ESC-" in response.reply or any("8.1" in c or "8.2" in c for c in response.citations)


def test_agent_policy_question_citations(agent):
    response = agent.chat("What is the waiting period for maternity and childbirth?", provider_override="mock")
    assert "10-month" in response.reply.lower() or "maternity" in response.reply.lower()
    assert any("3.3" in c for c in response.citations)


# 5. FastAPI Endpoints Tests
def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["policy_chunks_indexed"] > 0
    assert data["claims_loaded"] == 6


def test_api_chat(client):
    payload = {
        "message": "What is the waiting period for pre-existing conditions?",
        "provider": "mock"
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["citations"]) > 0


def test_api_get_claim(client):
    res = client.get("/api/claims/CLM1001")
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == "CLM1001"
    assert data["status"] == "Approved"


def test_api_escalate_claim(client):
    payload = {"reason": "Customer requests live review of bill reduction"}
    res = client.post("/api/claims/CLM1003/escalate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == "CLM1003"
    assert data["status"] == "Escalated to Specialist"


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


# 6. Advanced Algorithm Tests
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
    acc = BenefitAccumulator()
    # First claim: $800 bill, deductible is $1000
    res1 = acc.adjudicate_and_accumulate("POL-TEST", 800.0, in_network=True)
    assert res1["deductible_applied"] == 800.0
    assert res1["deductible_remaining_after"] == 200.0
    assert res1["plan_payout"] == 0.0

    # Second claim: $1000 bill. Remaining deductible is $200.
    # $200 goes to deductible. $800 subject to 80% coinsurance = $640 plan payout, $160 patient coinsurance.
    res2 = acc.adjudicate_and_accumulate("POL-TEST", 1000.0, in_network=True)
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
    # Test cosmetic exclusion
    res_cosmetic = claims_service.adjudicate_pre_claim("Cosmetic Rhinoplasty", 4000.0)
    assert res_cosmetic["decision"] == "Excluded"
    assert res_cosmetic["is_eligible"] is False

    # Test covered emergency room
    res_er = claims_service.adjudicate_pre_claim("Emergency Appendix Surgery", 3000.0)
    assert res_er["decision"] == "Covered"
    assert res_er["is_eligible"] is True


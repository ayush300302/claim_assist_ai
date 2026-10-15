"""FastAPI backend application for ClaimAssist AI."""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval import PolicyRetriever, PolicyChunk, SearchResult
from app.claims_service import ClaimsService, ClaimRecord, EscalationTicket
from app.audit import AuditLogger
from app.agent import ClaimAssistAgent, AgentMessage, AgentResponse


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Insurance Claim Adjudication, Retrieval, and Specialist Escalation API"
)

# Enable CORS for Streamlit and external frontend web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize application components
retriever = PolicyRetriever()
claims_service = ClaimsService()
audit_logger = AuditLogger()
agent = ClaimAssistAgent(retriever=retriever, claims_service=claims_service, audit_logger=audit_logger)


# Request & Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or inquiry text")
    history: List[AgentMessage] = Field(default_factory=list, description="Prior conversation messages")
    session_id: str = Field(default="session-web-1", description="Session tracking identifier")
    provider: Optional[str] = Field(default=None, description="Override LLM provider ('openrouter', 'openai', 'mock')")
    api_key: Optional[str] = Field(default=None, description="Optional dynamic API key override")
    model: Optional[str] = Field(default=None, description="Optional model name override")


class EscalateRequest(BaseModel):
    reason: str = Field(..., description="Detailed explanation for escalating the claim")
    policyholder_id: Optional[str] = Field(default=None, description="Associated policyholder ID")


class ReimbursementRequest(BaseModel):
    incurred_charges: float = Field(..., gt=0, description="Total medical expense incurred")
    deductible_remaining: float = Field(default=0.0, ge=0, description="Remaining annual deductible")
    non_covered_charges: float = Field(default=0.0, ge=0, description="Charges for non-covered items")
    in_network: bool = Field(default=True, description="Whether treatment was rendered in-network")


class PolicySearchRequest(BaseModel):
    query: str = Field(..., description="Keyword or semantic search phrase")
    top_k: int = Field(default=3, ge=1, le=10, description="Max chunk results to return")


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "configured_provider": settings.LLM_PROVIDER,
        "default_openrouter_model": settings.OPENROUTER_MODEL
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "provider": settings.LLM_PROVIDER,
        "policy_chunks_indexed": len(retriever.get_all_chunks()),
        "claims_loaded": len(claims_service.list_all_claims())
    }


@app.post("/api/chat", response_model=AgentResponse)
def chat_endpoint(payload: ChatRequest):
    """Primary chat endpoint connecting user inquiries with the ClaimAssist AI agent."""
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    response = agent.chat(
        user_query=payload.message,
        conversation_history=payload.history,
        session_id=payload.session_id,
        provider_override=payload.provider,
        api_key_override=payload.api_key,
        model_override=payload.model
    )
    return response


@app.get("/api/claims", response_model=List[ClaimRecord])
def list_claims(search: Optional[str] = None):
    """Retrieve claims registry, with optional search filter."""
    if search:
        return claims_service.search_claims(search)
    return claims_service.list_all_claims()


@app.get("/api/claims/{claim_id}", response_model=ClaimRecord)
def get_claim_details(claim_id: str):
    """Retrieve complete claim details for a specific Claim ID."""
    claim = claims_service.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim with ID '{claim_id}' not found."
        )
    return claim


@app.post("/api/claims/{claim_id}/escalate", response_model=EscalationTicket)
def escalate_claim_endpoint(claim_id: str, payload: EscalateRequest):
    """Escalate a disputed or delayed claim to a licensed human specialist per Section 8.2."""
    claim = claims_service.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cannot escalate: Claim '{claim_id}' was not found in records."
        )
    return claims_service.escalate_claim(
        claim_id=claim_id,
        reason=payload.reason,
        policyholder_id=payload.policyholder_id or claim.policyholder_id
    )


@app.post("/api/reimbursement/calculate")
def calculate_reimbursement_endpoint(payload: ReimbursementRequest):
    """Calculate reimbursement estimation according to Section 5.3."""
    return claims_service.calculate_reimbursement(
        incurred_charges=payload.incurred_charges,
        deductible_remaining=payload.deductible_remaining,
        non_covered_charges=payload.non_covered_charges,
        in_network=payload.in_network
    )


@app.post("/api/policy/search")
def search_policy_endpoint(payload: PolicySearchRequest):
    """Search policy document sections."""
    results = retriever.search(payload.query, top_k=payload.top_k)
    return [
        {
            "section_id": r.chunk.section_id,
            "section_title": r.chunk.section_title,
            "category": r.chunk.category,
            "citation": r.chunk.citation,
            "content": r.chunk.content,
            "score": r.score
        }
        for r in results
    ]


@app.get("/api/policy/sections")
def list_policy_sections():
    """Retrieve full catalog of policy sections and citations."""
    chunks = retriever.get_all_chunks()
    return [
        {
            "section_id": c.section_id,
            "section_title": c.section_title,
            "category": c.category,
            "citation": c.citation,
            "content": c.content
        }
        for c in chunks
    ]


@app.get("/api/claims/{claim_id}/anomaly")
def check_claim_anomaly_endpoint(claim_id: str):
    """Run Z-score anomaly detection and Section 5.4 timely filing audit on a claim."""
    claim = claims_service.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    from app.algorithms.fraud_detector import ClaimAnomalyDetector
    return ClaimAnomalyDetector.evaluate_claim(
        claim_id=claim.claim_id,
        claim_type=claim.claim_type,
        claim_amount=claim.claim_amount,
        submission_date=claim.submission_date
    )


@app.post("/api/claims/{claim_id}/appeal-letter")
def generate_appeal_letter_endpoint(claim_id: str, dispute_grounds: Optional[str] = None):
    """Generate formal legal dispute and appeal letter per Section 8.1."""
    try:
        return claims_service.generate_appeal_letter(claim_id, dispute_grounds)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class PreClaimRequest(BaseModel):
    treatment_name: str
    estimated_cost: float
    months_enrolled: int = 12
    is_pre_existing: bool = False
    in_network: bool = True


@app.post("/api/claims/adjudicate-pre-claim")
def adjudicate_pre_claim_endpoint(payload: PreClaimRequest):
    """Simulate pre-claim adjudication checking exclusions, waiting periods, and deductible rules."""
    return claims_service.adjudicate_pre_claim(
        treatment_name=payload.treatment_name,
        estimated_cost=payload.estimated_cost,
        months_enrolled=payload.months_enrolled,
        is_pre_existing=payload.is_pre_existing,
        in_network=payload.in_network
    )


class AccumulatorRequest(BaseModel):
    policyholder_id: str
    eligible_charges: float
    in_network: bool = True


@app.post("/api/accumulator/adjudicate")
def accumulate_benefits_endpoint(payload: AccumulatorRequest):
    """Adjudicate claim through stateful deductible & Out-Of-Pocket Maximum accumulator."""
    return agent.accumulator.adjudicate_and_accumulate(
        policyholder_id=payload.policyholder_id,
        eligible_charges=payload.eligible_charges,
        in_network=payload.in_network
    )


@app.get("/api/audit/logs")
def get_audit_logs(limit: int = Query(default=20, ge=1, le=100)):
    """Retrieve recent conversation audit log entries."""
    return audit_logger.read_recent_logs(limit=limit)


"""Claims lookup, validation, calculation, and escalation service."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.config import settings


class ClaimRecord(BaseModel):
    """Schema representing an official insurance claim record."""
    claim_id: str
    policyholder_id: str
    policyholder_name: str
    claim_type: str
    claim_amount: float
    approved_amount: float
    status: str
    submission_date: str
    last_updated: str
    next_action: str
    notes: str


class EscalationTicket(BaseModel):
    """Schema representing an escalation request to a licensed human claims specialist."""
    ticket_id: str = Field(default_factory=lambda: f"ESC-{uuid.uuid4().hex[:6].upper()}")
    claim_id: str
    policyholder_id: Optional[str] = None
    reason: str
    status: str = "Escalated to Specialist"
    assigned_queue: str = "Human Adjudication & Appeals Unit"
    specialist_hotline: str = "1-800-555-APEX (1-800-555-2739), Mon–Fri 8am-6pm EST"
    specialist_email: str = "appeals@apexshield-care.example.com"
    turnaround_notice: str = "A senior specialist will review your file within 2 business days."
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ClaimsService:
    """Service handling claims querying, calculations, and specialist escalations."""

    def __init__(self, claims_path: Optional[Path] = None):
        self.claims_path = claims_path or settings.resolved_claims_path
        self._claims_cache: Dict[str, ClaimRecord] = {}
        self._escalations: List[EscalationTicket] = []
        self._load_claims()

    def _load_claims(self) -> None:
        """Loads claims data from JSON storage."""
        if not self.claims_path.exists():
            raise FileNotFoundError(f"Claims data file not found at {self.claims_path}")

        raw_data = json.loads(self.claims_path.read_text(encoding="utf-8"))
        self._claims_cache = {
            item["claim_id"].upper(): ClaimRecord(**item)
            for item in raw_data
        }

    def get_claim(self, claim_id: str) -> Optional[ClaimRecord]:
        """Look up a claim record by its unique ID (e.g., 'CLM1001')."""
        normalized = claim_id.strip().upper()
        return self._claims_cache.get(normalized)

    def search_claims(self, query: str) -> List[ClaimRecord]:
        """Search claims by Claim ID, Policyholder ID, or Policyholder Name."""
        query_norm = query.strip().lower()
        results: List[ClaimRecord] = []
        for claim in self._claims_cache.values():
            if (query_norm in claim.claim_id.lower() or
                query_norm in claim.policyholder_id.lower() or
                query_norm in claim.policyholder_name.lower() or
                query_norm in claim.claim_type.lower() or
                query_norm in claim.status.lower()):
                results.append(claim)
        return results

    def list_all_claims(self) -> List[ClaimRecord]:
        """Returns all claims records."""
        return list(self._claims_cache.values())

    def escalate_claim(self, claim_id: str, reason: str, policyholder_id: Optional[str] = None) -> EscalationTicket:
        """Escalates a claim dispute or delay to a human specialist per Section 8.2."""
        claim = self.get_claim(claim_id)
        pol_id = policyholder_id or (claim.policyholder_id if claim else "POL-UNKNOWN")

        ticket = EscalationTicket(
            claim_id=claim_id.upper(),
            policyholder_id=pol_id,
            reason=reason
        )
        self._escalations.append(ticket)
        return ticket

    def calculate_reimbursement(
        self,
        incurred_charges: float,
        deductible_remaining: float = 0.0,
        non_covered_charges: float = 0.0,
        in_network: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate reimbursement breakdown per Section 5.3:
        Reimbursement = (Eligible Incurred Charges - Applicable Deductible - Non-Covered Line Items) * Co-Insurance Rate
        """
        co_insurance_rate = 0.80 if in_network else 0.70
        eligible_amount = max(0.0, incurred_charges - non_covered_charges)
        amount_subject_to_coinsurance = max(0.0, eligible_amount - deductible_remaining)
        estimated_reimbursement = round(amount_subject_to_coinsurance * co_insurance_rate, 2)
        patient_responsibility = round(incurred_charges - estimated_reimbursement, 2)

        return {
            "incurred_charges": round(incurred_charges, 2),
            "non_covered_charges": round(non_covered_charges, 2),
            "deductible_applied": round(min(eligible_amount, deductible_remaining), 2),
            "co_insurance_rate": f"{int(co_insurance_rate * 100)}%",
            "network_status": "In-Network (80%)" if in_network else "Out-of-Network (70%)",
            "estimated_reimbursement": estimated_reimbursement,
            "patient_estimated_responsibility": patient_responsibility,
            "governing_section": "[Section 5.3] Reimbursement Calculation"
        }

    def generate_appeal_letter(self, claim_id: str, dispute_grounds: Optional[str] = None) -> Dict[str, Any]:
        """Generates a formal, legally grounded insurance appeal packet per Section 8.1."""
        claim = self.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim ID '{claim_id}' not found.")

        today_str = datetime.now().strftime("%B %d, %Y")
        grounds = dispute_grounds or (
            f"The policyholder disputes the adjudication decision for {claim.claim_type}. "
            f"Official records indicate medical necessity and compliance with policy terms."
        )

        letter_body = f"""FORMAL NOTICE OF CLAIM DISPUTE & APPEAL
Pursuant to ApexShield Care Plus Policy [Section 8.1] & [Section 8.2]

Date: {today_str}
To: ApexShield Claims Appeals & Clinical Adjudication Unit
Email: appeals@apexshield-care.example.com
Phone: 1-800-555-APEX (1-800-555-2739)

RE: FORMAL FIRST-LEVEL APPEAL FOR CLAIM {claim.claim_id}
Policyholder Name: {claim.policyholder_name}
Policyholder ID: {claim.policyholder_id}
Claim Type: {claim.claim_type}
Total Incurred Amount: ${claim.claim_amount:,.2f}
Initial Adjudication Status: {claim.status}
Original Submission Date: {claim.submission_date}

Dear Appeals Committee,

I am writing to formally submit a First-Level Dispute and Appeal regarding the adjudication of Claim {claim.claim_id}, originally filed on {claim.submission_date}.

CURRENT RECORD AND FINDINGS:
According to the Explanation of Benefits (EOB) and plan notes:
"{claim.notes}"

GROUNDS FOR APPEAL:
{grounds}

REQUESTED ACTION & CITATIONS:
1. In accordance with [Section 8.1], this appeal is timely submitted within the 60 calendar-day dispute window from the Explanation of Benefits date.
2. Under [Section 8.2], I request that this file be reassigned to a licensed Human Claims Specialist and Clinical Adjudication Peer for comprehensive medical necessity re-examination.
3. If additional itemized clinical summaries or medical records (per Section 6) are needed, please provide formal notice via appeals@apexshield-care.example.com.

Thank you for your prompt re-examination. I look forward to receiving your written determination within the standard adjudication turnaround timeframe.

Sincerely,

{claim.policyholder_name}
Policy ID: {claim.policyholder_id}
Claim ID: {claim.claim_id}
"""
        return {
            "claim_id": claim.claim_id,
            "policyholder_name": claim.policyholder_name,
            "policyholder_id": claim.policyholder_id,
            "appeal_channel": "appeals@apexshield-care.example.com",
            "statutory_deadline": "60 calendar days from EOB date (Section 8.1)",
            "letter_content": letter_body.strip()
        }

    def adjudicate_pre_claim(
        self,
        treatment_name: str,
        estimated_cost: float,
        months_enrolled: int = 12,
        is_pre_existing: bool = False,
        in_network: bool = True
    ) -> Dict[str, Any]:
        """
        Simulates pre-claim adjudication against policy terms, waiting periods, and exclusions.
        """
        lower_tx = treatment_name.lower()
        decision = "Covered"
        governing_sections = ["[Section 2] Coverage Conditions"]
        notes = []

        # Check Exclusions (Section 4)
        if any(w in lower_tx for w in ["cosmetic", "rhinoplasty", "aesthetic", "hair transplant", "contouring"]):
            decision = "Excluded"
            governing_sections.append("[Section 4.1] Cosmetic & Elective Surgery Exclusion")
            notes.append("Elective cosmetic procedures without accidental injury reconstructive need are excluded.")
        elif any(w in lower_tx for w in ["experimental", "clinical trial"]):
            decision = "Excluded"
            governing_sections.append("[Section 4.2] Experimental or Unproven Therapies")
            notes.append("Unapproved or experimental therapies are strictly excluded.")
        elif any(w in lower_tx for w in ["dental", "teeth", "braces", "lasik", "vision"]):
            decision = "Excluded"
            governing_sections.append("[Section 4.5] Dental and Vision Routine Care")
            notes.append("Routine dental and vision procedures are excluded under standard Care Plus.")

        # Check Waiting Periods (Section 3) if not excluded
        if decision != "Excluded":
            if months_enrolled < 1 and "emergency" not in lower_tx and "accident" not in lower_tx:
                decision = "Waiting Period Active"
                governing_sections.append("[Section 3.1] General Waiting Period (30 Days)")
                notes.append("30-day general waiting period applies to non-emergency treatment.")

            if is_pre_existing and months_enrolled < 12:
                decision = "Waiting Period Active"
                governing_sections.append("[Section 3.2] Pre-Existing Conditions (12 Months)")
                notes.append(f"Pre-existing conditions require 12 consecutive months of coverage. (Enrolled: {months_enrolled} mos).")

            if any(w in lower_tx for w in ["maternity", "childbirth", "delivery", "cesarean"]) and months_enrolled < 10:
                decision = "Waiting Period Active"
                governing_sections.append("[Section 3.3] Maternity & Childbirth (10 Months)")
                notes.append("Maternity requires a 10-month continuous coverage waiting period.")

            if any(w in lower_tx for w in ["joint replacement", "hernia", "cataract", "gallbladder"]) and months_enrolled < 3:
                decision = "Waiting Period Active"
                governing_sections.append("[Section 3.4] Specific Surgeries (90 Days)")
                notes.append("Specific elective surgeries require a 90-day waiting period.")

        # Pre-authorization check (Section 2.3)
        if any(w in lower_tx for w in ["mri", "pet", "ct scan"]) and estimated_cost > 500:
            governing_sections.append("[Section 2.3] Diagnostic Tests & Imaging Pre-Authorization")
            notes.append("Advanced imaging exceeding $500 requires pre-authorization on file.")

        # Payout calculation
        reimbursement = self.calculate_reimbursement(
            incurred_charges=estimated_cost,
            deductible_remaining=0.0,
            non_covered_charges=estimated_cost if decision in ["Excluded", "Waiting Period Active"] else 0.0,
            in_network=in_network
        )

        return {
            "treatment_name": treatment_name,
            "estimated_cost": estimated_cost,
            "decision": decision,
            "is_eligible": decision == "Covered",
            "governing_sections": list(dict.fromkeys(governing_sections)),
            "notes": " ".join(notes) if notes else "Eligible for reimbursement subject to standard deductible and co-insurance.",
            "estimated_reimbursement": reimbursement["estimated_reimbursement"] if decision == "Covered" else 0.0,
            "patient_estimated_share": reimbursement["patient_estimated_responsibility"] if decision == "Covered" else estimated_cost,
            "required_documents": [
                "Form CL-01 Claim Form",
                "Itemized hospital/clinic bill with procedure codes",
                "Attending physician diagnosis / discharge summary (ICD-10)"
            ]
        }

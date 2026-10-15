"""Algorithmic Claim Anomaly & Fraud Detection Engine."""

import math
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


# Historical benchmark statistical distributions for healthcare claim categories (Mean, StdDev)
BENCHMARK_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "Emergency Room": {"mean": 1200.0, "std": 450.0, "max_normal": 3500.0},
    "Diagnostic Imaging": {"mean": 750.0, "std": 300.0, "max_normal": 2200.0},
    "Inpatient Surgery": {"mean": 5800.0, "std": 2100.0, "max_normal": 15000.0},
    "Outpatient Specialist": {"mean": 240.0, "std": 80.0, "max_normal": 800.0},
    "Preventive Care": {"mean": 320.0, "std": 100.0, "max_normal": 900.0},
    "Cosmetic Rhinoplasty": {"mean": 4200.0, "std": 1100.0, "max_normal": 9000.0}
}


class AnomalyReport(BaseModel):
    claim_id: str
    risk_score: float  # 0.0 (Clean) to 1.0 (Critical Risk)
    risk_level: str   # "LOW", "MEDIUM", "HIGH"
    z_score: float
    timely_filing_ok: bool
    days_elapsed: Optional[int] = None
    flags: List[str]
    recommendation: str


class ClaimAnomalyDetector:
    """
    Statistical Z-Score & Timely Filing Compliance Anomaly Detector for Insurance Claims.
    """

    @staticmethod
    def calculate_z_score(amount: float, claim_type: str) -> float:
        stats = BENCHMARK_DISTRIBUTIONS.get(claim_type, {"mean": 1500.0, "std": 800.0})
        z = (amount - stats["mean"]) / max(stats["std"], 1.0)
        return round(float(z), 2)

    @classmethod
    def evaluate_claim(
        cls,
        claim_id: str,
        claim_type: str,
        claim_amount: float,
        submission_date: str,
        service_date: Optional[str] = None
    ) -> AnomalyReport:
        flags: List[str] = []
        z = cls.calculate_z_score(claim_amount, claim_type)
        stats = BENCHMARK_DISTRIBUTIONS.get(claim_type, {"mean": 1500.0, "std": 800.0, "max_normal": 5000.0})

        # 1. Statistical Outlier Detection (Z-Score > 2.5 is statistically rare outlier)
        if z > 2.5:
            flags.append(f"Statistically anomalous amount: Z-Score={z} (+{z:.1f}σ above benchmark mean ${stats['mean']:,.0f})")
        elif z > 1.8:
            flags.append(f"Elevated claim amount (+{z:.1f}σ above category average)")

        # 2. Timely Filing Rule (Section 5.4: 90 days)
        days_elapsed = None
        timely_filing_ok = True
        if service_date:
            try:
                dt_serv = datetime.strptime(service_date, "%Y-%m-%d")
                dt_sub = datetime.strptime(submission_date, "%Y-%m-%d")
                days_elapsed = (dt_sub - dt_serv).days
                if days_elapsed > 90:
                    timely_filing_ok = False
                    flags.append(f"Filing deadline violation: {days_elapsed} days elapsed since service (> 90-day limit per [Section 5.4])")
            except ValueError:
                pass

        # 3. Known Non-Covered / Cosmetic Procedure Flag
        if "cosmetic" in claim_type.lower() or "rhinoplasty" in claim_type.lower():
            flags.append("Procedure matches Section 4.1 Cosmetic Surgery Exclusion criteria")

        # 4. Composite Risk Score Algorithm
        risk_score = 0.1
        if z > 2.5:
            risk_score += 0.45
        elif z > 1.8:
            risk_score += 0.25

        if not timely_filing_ok:
            risk_score += 0.40

        if any("cosmetic" in f.lower() for f in flags):
            risk_score += 0.35

        risk_score = min(1.0, round(risk_score, 2))

        if risk_score >= 0.65:
            risk_level = "HIGH"
            rec = "Route immediately to Clinical SIU / Senior Fraud Adjudication review prior to disbursement."
        elif risk_score >= 0.35:
            risk_level = "MEDIUM"
            rec = "Secondary documentation check required (Section 6 itemized bill & physician notes)."
        else:
            risk_level = "LOW"
            rec = "Standard automated adjudication queue (clean claim)."

        return AnomalyReport(
            claim_id=claim_id,
            risk_score=risk_score,
            risk_level=risk_level,
            z_score=z,
            timely_filing_ok=timely_filing_ok,
            days_elapsed=days_elapsed,
            flags=flags,
            recommendation=rec
        )

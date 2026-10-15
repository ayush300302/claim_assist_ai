"""Algorithmic Faithfulness & Hallucination Guardrail."""

import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel


class GroundingAudit(BaseModel):
    is_faithful: bool
    grounding_score: float  # 0.0 to 1.0 (1.0 = highly grounded)
    cited_sections: List[str]
    valid_citations: List[str]
    invalid_citations: List[str]
    overlap_ratio: float
    warning_flag: bool
    audit_notes: str


class FaithfulnessGuardrail:
    """
    Evaluates whether an assistant's response is factually grounded in the retrieved policy text
    and ensures all cited section numbers match verified policy chapters.
    """

    KNOWN_VALID_SECTIONS = {
        "1", "2", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
        "3", "3.1", "3.2", "3.3", "3.4", "3.5",
        "4", "4.1", "4.2", "4.3", "4.4", "4.5",
        "5", "5.1", "5.2", "5.3", "5.4",
        "6",
        "7", "7.1", "7.2", "7.3", "7.4",
        "8", "8.1", "8.2", "8.3"
    }

    @classmethod
    def evaluate(
        cls,
        assistant_reply: str,
        retrieved_context: str
    ) -> GroundingAudit:
        # 1. Extract all cited sections, e.g., "[Section 2.1]" or "Section 3.2"
        found_cites = re.findall(r"Section\s+([0-9]+(?:\.[0-9]+)?)", assistant_reply, re.IGNORECASE)
        found_cites_norm = list(dict.fromkeys(found_cites))

        valid = [s for s in found_cites_norm if s in cls.KNOWN_VALID_SECTIONS]
        invalid = [s for s in found_cites_norm if s not in cls.KNOWN_VALID_SECTIONS]

        # 2. Extract significant factual words (length >= 4) from response and context
        reply_words = set(re.findall(r"\b[a-z]{4,}\b", assistant_reply.lower()))
        context_words = set(re.findall(r"\b[a-z]{4,}\b", retrieved_context.lower()))

        # Remove generic conversational stop words
        common_words = {
            "this", "that", "with", "from", "your", "have", "been", "under",
            "please", "claim", "policy", "apexshield", "assistant", "support",
            "status", "would", "could", "should", "their", "about", "which"
        }
        filtered_reply_words = reply_words - common_words

        if filtered_reply_words:
            overlap = filtered_reply_words.intersection(context_words)
            overlap_ratio = len(overlap) / len(filtered_reply_words)
        else:
            overlap_ratio = 1.0

        # 3. Grounding Confidence Calculation
        # Penalize if cited sections don't exist in policy
        citation_penalty = len(invalid) * 0.35
        base_score = overlap_ratio * 0.75 + (1.0 if valid else 0.4) * 0.25
        grounding_score = max(0.0, min(1.0, round(base_score - citation_penalty, 3)))

        is_faithful = grounding_score >= 0.50 and len(invalid) == 0
        warning = not is_faithful

        if invalid:
            notes = f"Warning: Hallucinated policy sections detected: {invalid}."
        elif grounding_score < 0.50:
            notes = "Low grounding confidence: response contains assertions not verified in retrieved excerpts."
        else:
            notes = "Verified: strictly grounded against official policy documentation."

        return GroundingAudit(
            is_faithful=is_faithful,
            grounding_score=grounding_score,
            cited_sections=[f"[Section {s}]" for s in found_cites_norm],
            valid_citations=[f"[Section {s}]" for s in valid],
            invalid_citations=[f"[Section {s}]" for s in invalid],
            overlap_ratio=round(overlap_ratio, 3),
            warning_flag=warning,
            audit_notes=notes
        )

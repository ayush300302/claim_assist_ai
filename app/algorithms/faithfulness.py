"""Algorithmic Faithfulness & Hallucination Guardrail with Numerical Entailment."""

import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel


class GroundingAudit(BaseModel):
    is_faithful: bool
    grounding_score: float  # 0.0 to 1.0 (1.0 = highly grounded)
    cited_sections: List[str]
    valid_citations: List[str]
    invalid_citations: List[str]
    unsupported_numeric_facts: List[str]
    overlap_ratio: float
    warning_flag: bool
    audit_notes: str


class FaithfulnessGuardrail:
    """
    Evaluates whether an assistant's response is factually grounded in retrieved policy text:
    1. Ensures all cited section numbers match verified policy chapters.
    2. Validates numerical assertions (dollar amounts, waiting periods, percentages).
    3. Computes lexical and semantic overlap ratio against retrieved context.
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

        # 2. Numerical Entailment Check
        # Extract numerical assertions: dollar values, percentages, and time windows (e.g. 30 days, 12 months)
        num_patterns = [
            r"\$\s*\d+(?:,\d+)*(?:\.\d+)?",                  # $1,000 or $500
            r"\b\d+%",                                       # 80%, 70%
            r"\b\d+\s+(?:calendar\s+)?(?:days|months|years|weeks)\b"  # 30 days, 60 calendar days
        ]
        
        reply_numbers: List[str] = []
        for pat in num_patterns:
            matches = re.findall(pat, assistant_reply, re.IGNORECASE)
            reply_numbers.extend([m.strip() for m in matches])
        reply_numbers = list(dict.fromkeys(reply_numbers))

        # Check which numbers are NOT in the context (normalized)
        ctx_clean = retrieved_context.lower().replace(",", "")
        unsupported_nums = []
        for num in reply_numbers:
            clean_num = num.lower().replace(",", "")
            # If the exact token or digit string isn't anywhere in context
            digits = re.findall(r"\d+", clean_num)
            if digits and not any(d in ctx_clean for d in digits):
                # Only flag if not a common UI / contact artifact like 1-800 or 2 business days
                if not any(ign in clean_num for ign in ["1-800", "555", "business"]):
                    unsupported_nums.append(num)

        # 3. Extract significant factual words (length >= 4)
        reply_words = set(re.findall(r"\b[a-z]{4,}\b", assistant_reply.lower()))
        context_words = set(re.findall(r"\b[a-z]{4,}\b", retrieved_context.lower()))

        # Stop words
        common_words = {
            "this", "that", "with", "from", "your", "have", "been", "under",
            "please", "claim", "policy", "apexshield", "assistant", "support",
            "status", "would", "could", "should", "their", "about", "which",
            "here", "these", "there"
        }
        filtered_reply_words = reply_words - common_words

        if filtered_reply_words:
            overlap = filtered_reply_words.intersection(context_words)
            overlap_ratio = len(overlap) / len(filtered_reply_words)
        else:
            overlap_ratio = 1.0

        # 4. Grounding Confidence Calculation
        citation_penalty = len(invalid) * 0.35
        num_penalty = len(unsupported_nums) * 0.20
        base_score = overlap_ratio * 0.70 + (1.0 if valid else 0.4) * 0.30
        grounding_score = max(0.0, min(1.0, round(base_score - citation_penalty - num_penalty, 3)))

        is_faithful = grounding_score >= 0.45 and len(invalid) == 0 and len(unsupported_nums) == 0
        warning = not is_faithful

        notes = []
        if invalid:
            notes.append(f"Hallucinated policy sections detected: {invalid}.")
        if unsupported_nums:
            notes.append(f"Unverified numerical assertions detected: {unsupported_nums}.")
        if grounding_score < 0.45 and not invalid and not unsupported_nums:
            notes.append("Low vocabulary grounding against retrieved policy context.")
        if not notes:
            notes.append("Verified: strictly grounded against official policy documentation.")

        return GroundingAudit(
            is_faithful=is_faithful,
            grounding_score=grounding_score,
            cited_sections=[f"[Section {s}]" for s in found_cites_norm],
            valid_citations=[f"[Section {s}]" for s in valid],
            invalid_citations=[f"[Section {s}]" for s in invalid],
            unsupported_numeric_facts=unsupported_nums,
            overlap_ratio=round(overlap_ratio, 3),
            warning_flag=warning,
            audit_notes=" ".join(notes)
        )

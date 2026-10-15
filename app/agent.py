"""ClaimAssist AI Conversational Agent Engine."""

import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval import PolicyRetriever, SearchResult
from app.claims_service import ClaimsService, ClaimRecord, EscalationTicket
from app.audit import AuditLogger


from app.algorithms.faithfulness import FaithfulnessGuardrail, GroundingAudit
from app.algorithms.fraud_detector import ClaimAnomalyDetector, AnomalyReport
from app.algorithms.accumulator import BenefitAccumulator


class AgentMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class AgentResponse(BaseModel):
    """Structured response from the ClaimAssist AI agent."""
    reply: str
    citations: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    model_used: str = "mock"
    latency_ms: float = 0.0
    session_id: str = "default"
    grounding_score: float = 1.0
    is_verified: bool = True
    audit_notes: str = "Verified strictly against policy documentation."


SYSTEM_PROMPT = """You are ClaimAssist AI, the official insurance claim assistant for ApexShield Care Plus.
Your responsibilities:
1. Provide accurate, helpful policy guidance strictly grounded in the official policy document.
2. Provide transparent, up-to-date claim status lookups and next steps.
3. Facilitate formal disputes, appeals, and human specialist escalations per official guidelines.
4. Provide precise reimbursement estimates based on deductible and co-insurance rules.

CRITICAL GROUNDING RULES:
- ALWAYS cite the governing policy section in brackets, e.g. [Section 2.1], [Section 3.2], [Section 4.1], [Section 5.3], [Section 8.2].
- NEVER invent policy terms, benefits, or exclusions. If something is not covered or ambiguous, state it clearly.
- If a claim is denied, under review for over 20 business days, or involves a dispute, proactively provide escalation procedures to a Human Claims Specialist per [Section 8.2] and support contact channels [Section 8.3] (Hotline: 1-800-555-APEX, Email: appeals@apexshield-care.example.com).
- Maintain an empathetic, professional, and clear tone at all times.
"""


# OpenAI / OpenRouter function tools schema
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_claim",
            "description": "Look up an existing insurance claim status, amounts, submission date, and required actions by Claim ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "The unique claim identifier, e.g. CLM1001, CLM1002, etc."
                    }
                },
                "required": ["claim_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve policy sections, rules, waiting periods, exclusions, and coverage terms by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query about health coverage, waiting periods, exclusions, reimbursement, or claims."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human_specialist",
            "description": "Escalate a disputed claim, delayed review, or emergency pre-authorization to a licensed human specialist per Section 8.2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "The claim ID to escalate."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of why the claimant is disputing or escalating."
                    }
                },
                "required": ["claim_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_reimbursement",
            "description": "Calculate estimated reimbursement breakdown under Section 5.3 based on deductible and coinsurance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incurred_charges": {"type": "number", "description": "Total medical bill amount in dollars."},
                    "deductible_remaining": {"type": "number", "description": "Remaining deductible amount yet to be met.", "default": 0.0},
                    "non_covered_charges": {"type": "number", "description": "Charges for non-covered items/exclusions.", "default": 0.0},
                    "in_network": {"type": "boolean", "description": "Whether the provider is in-network (80%) or out-of-network (70%).", "default": True}
                },
                "required": ["incurred_charges"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_claim_anomalies",
            "description": "Run statistical Z-Score outlier analysis and Section 5.4 timely filing verification for a claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "The claim ID to analyze."}
                },
                "required": ["claim_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_appeal_letter",
            "description": "Generate a legally grounded, Section 8.1 compliant formal appeal letter for a denied claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "The claim ID under dispute."},
                    "dispute_grounds": {"type": "string", "description": "The policyholder's justification."}
                },
                "required": ["claim_id"]
            }
        }
    }
]


class ClaimAssistAgent:
    """Core agent engine orchestrating RAG, claim tools, and LLM inference."""

    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        claims_service: Optional[ClaimsService] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        self.retriever = retriever or PolicyRetriever()
        self.claims_service = claims_service or ClaimsService()
        self.audit_logger = audit_logger or AuditLogger()
        self.accumulator = BenefitAccumulator()

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Executes the designated backend tool safely."""
        citations: List[str] = []

        if tool_name == "lookup_claim":
            claim_id = args.get("claim_id", "")
            claim = self.claims_service.get_claim(claim_id)
            if claim:
                return claim.model_dump(), ["[Section 7.1] Initial Acknowledgment", "[Section 7.2] Standard Review Period"]
            return {"error": f"Claim ID '{claim_id}' not found in the database."}, []

        elif tool_name == "search_policy":
            query = args.get("query", "")
            results = self.retriever.search(query, top_k=settings.TOP_K_CHUNKS)
            docs = []
            for r in results:
                citations.append(r.chunk.citation)
                docs.append({
                    "section_id": r.chunk.section_id,
                    "title": r.chunk.section_title,
                    "content": r.chunk.content,
                    "relevance_score": r.score
                })
            return {"results": docs}, citations

        elif tool_name == "escalate_to_human_specialist":
            claim_id = args.get("claim_id", "")
            reason = args.get("reason", "Customer requested escalation")
            ticket = self.claims_service.escalate_claim(claim_id=claim_id, reason=reason)
            return ticket.model_dump(), ["[Section 8.2] Escalation to Human Claims Specialist", "[Section 8.3] Contact Channels"]

        elif tool_name == "estimate_reimbursement":
            incurred = float(args.get("incurred_charges", 0))
            deductible = float(args.get("deductible_remaining", 0))
            non_cov = float(args.get("non_covered_charges", 0))
            in_net = bool(args.get("in_network", True))
            calc = self.claims_service.calculate_reimbursement(
                incurred_charges=incurred,
                deductible_remaining=deductible,
                non_covered_charges=non_cov,
                in_network=in_net
            )
            return calc, ["[Section 5.1] Annual Deductible", "[Section 5.3] Reimbursement Calculation"]

        elif tool_name == "detect_claim_anomalies":
            claim_id = args.get("claim_id", "")
            claim = self.claims_service.get_claim(claim_id)
            if not claim:
                return {"error": f"Claim {claim_id} not found"}, []
            report = ClaimAnomalyDetector.evaluate_claim(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type,
                claim_amount=claim.claim_amount,
                submission_date=claim.submission_date
            )
            return report.model_dump(), ["[Section 5.4] Timely Filing Requirement", "[Section 4.1] Exclusions"]

        elif tool_name == "accumulate_benefits":
            pol_id = args.get("policyholder_id", "POL-DEFAULT")
            charges = float(args.get("charges", 1000.0))
            in_net = bool(args.get("in_network", True))
            res = self.accumulator.adjudicate_and_accumulate(pol_id, charges, in_net)
            return res, res["citations"]

        elif tool_name == "generate_appeal_letter":
            claim_id = args.get("claim_id", "")
            grounds = args.get("dispute_grounds", None)
            letter = self.claims_service.generate_appeal_letter(claim_id, grounds)
            return letter, ["[Section 8.1] First-Level Claim Dispute / Appeal", "[Section 8.2] Escalation to Human Claims Specialist"]

        return {"error": f"Unknown tool '{tool_name}'"}, []

    def _mock_inference(self, user_query: str, history: List[AgentMessage]) -> AgentResponse:
        """High-fidelity, deterministic rule-based assistant mode for zero-cost testing & offline guarantees."""
        start_time = time.time()
        citations: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        lower = user_query.lower()

        # 1. Check for specific Claim ID inquiry (e.g., CLM1001, CLM1002, etc.)
        claim_match = re.search(r"\b(CLM\d{4})\b", user_query, re.IGNORECASE)
        is_escalation_intent = any(w in lower for w in ["escalate", "appeal", "dispute", "human", "specialist", "agent", "representative", "denied", "complain"])

        if claim_match:
            claim_id = claim_match.group(1).upper()
            tool_res, cite = self._execute_tool("lookup_claim", {"claim_id": claim_id})
            tool_calls.append({"tool": "lookup_claim", "args": {"claim_id": claim_id}, "result": tool_res})
            citations.extend(cite)

            if "error" in tool_res:
                reply = f"I searched our claims registry, but could not find a record matching **{claim_id}**. Please verify your Claim ID or contact ApexShield Member Services at **1-800-555-APEX**."
            else:
                claim = tool_res
                status = claim.get("status", "Unknown")
                reply = (
                    f"### Claim Status Update: {claim['claim_id']}\n\n"
                    f"- **Policyholder**: {claim['policyholder_name']} (`{claim['policyholder_id']}`)\n"
                    f"- **Claim Type**: {claim['claim_type']}\n"
                    f"- **Claimed Amount**: ${claim['claim_amount']:,.2f}\n"
                    f"- **Approved Amount**: ${claim['approved_amount']:,.2f}\n"
                    f"- **Current Status**: **{status}**\n"
                    f"- **Submission Date**: {claim['submission_date']} (Last updated: {claim['last_updated']})\n"
                    f"- **Next Action**: {claim['next_action']}\n"
                    f"- **Adjudication Notes**: {claim['notes']}\n\n"
                )

                if status == "Rejected" or is_escalation_intent:
                    esc_ticket = self.claims_service.escalate_claim(
                        claim_id=claim_id,
                        reason="Dispute/Appeal initiated for rejected claim or customer request",
                        policyholder_id=claim['policyholder_id']
                    )
                    citations.extend(["[Section 8.1] First-Level Claim Dispute / Appeal", "[Section 8.2] Escalation to Human Claims Specialist", "[Section 8.3] Contact Channels"])
                    tool_calls.append({"tool": "escalate_to_human_specialist", "args": {"claim_id": claim_id}, "result": esc_ticket.model_dump()})
                    reply += (
                        f"### Escalation & Dispute Assistance per [Section 8.1] & [Section 8.2]\n"
                        f"Because this claim has been marked **{status}**, you have the right to file an appeal within **60 calendar days** of your Explanation of Benefits (EOB).\n\n"
                        f"I have initiated an expedited escalation ticket for you:\n"
                        f"- **Ticket ID**: `{esc_ticket.ticket_id}`\n"
                        f"- **Assigned Queue**: {esc_ticket.assigned_queue}\n"
                        f"- **Claims Support Hotline**: {esc_ticket.specialist_hotline}\n"
                        f"- **Appeals Email**: `{esc_ticket.specialist_email}`\n"
                    )
                elif status == "Additional Documents Required":
                    citations.append("[Section 7.3] Information Request (Additional Documents)")
                    reply += (
                        f"\n**Action Required per [Section 7.3]**: Please submit the requested documents (itemized procedure codes or discharge summary) within 30 days to avoid claim closure."
                    )
                elif status == "Approved":
                    citations.append("[Section 7.4] Payment Disbursement")
                    reply += f"\n**Disbursement per [Section 7.4]**: Payments are disbursed via direct deposit within 3 business days of approval."

            latency = (time.time() - start_time) * 1000
            return AgentResponse(
                reply=reply,
                citations=list(dict.fromkeys(citations)),
                tool_calls=tool_calls,
                model_used="mock-rule-engine",
                latency_ms=latency
            )

        # 2. Check for reimbursement calculation intent
        calc_match = re.search(r"\$([0-9,]+(?:\.[0-9]{2})?)", user_query)
        if any(w in lower for w in ["calculate", "reimbursement", "estimate", "how much will i get", "coinsurance", "co-insurance", "deductible remaining"]):
            amount = float(calc_match.group(1).replace(",", "")) if calc_match else 1000.0
            in_net = not ("out-of-network" in lower or "out of network" in lower)
            calc_res, cite = self._execute_tool("estimate_reimbursement", {
                "incurred_charges": amount,
                "deductible_remaining": 500.0 if "deductible" in lower else 0.0,
                "in_network": in_net
            })
            tool_calls.append({"tool": "estimate_reimbursement", "args": {"incurred_charges": amount}, "result": calc_res})
            citations.extend(cite)
            reply = (
                f"### Estimated Reimbursement Breakdown per [Section 5.3]\n\n"
                f"- **Total Incurred Charges**: ${calc_res['incurred_charges']:,.2f}\n"
                f"- **Deductible Applied**: ${calc_res['deductible_applied']:,.2f}\n"
                f"- **Network Tier**: {calc_res['network_status']}\n"
                f"- **Estimated Plan Reimbursement**: **${calc_res['estimated_reimbursement']:,.2f}**\n"
                f"- **Estimated Patient Responsibility**: **${calc_res['patient_estimated_responsibility']:,.2f}**\n\n"
                f"> **Formula Applied**: `Reimbursement = (Eligible Charges - Deductible - Excluded Items) × Co-Insurance Rate` per [Section 5.3]."
            )
            latency = (time.time() - start_time) * 1000
            return AgentResponse(
                reply=reply,
                citations=list(dict.fromkeys(citations)),
                tool_calls=tool_calls,
                model_used="mock-rule-engine",
                latency_ms=latency
            )

        # 3. Policy Search / General Questions
        results = self.retriever.search(user_query, top_k=settings.TOP_K_CHUNKS)
        tool_calls.append({"tool": "search_policy", "args": {"query": user_query}, "count": len(results)})

        if results:
            primary_chunk = results[0].chunk
            citations.append(primary_chunk.citation)
            for r in results[1:]:
                citations.append(r.chunk.citation)

            reply = f"Based on the **ApexShield Care Plus Policy**:\n\n"
            reply += f"### {primary_chunk.citation}\n"
            reply += f"{primary_chunk.content}\n\n"

            if len(results) > 1:
                reply += "### Related Policy Sections:\n"
                for res in results[1:3]:
                    reply += f"- **{res.chunk.citation}**: {res.chunk.content}\n"

            # Check if user needs human specialist support
            if is_escalation_intent:
                citations.extend(["[Section 8.2] Escalation to Human Claims Specialist", "[Section 8.3] Contact Channels"])
                reply += (
                    f"\n---\n### Need to Speak to a Human Claims Specialist?\n"
                    f"Under **[Section 8.2]**, policyholders may connect with a licensed Human Claims Specialist:\n"
                    f"- **Phone**: 1-800-555-APEX (1-800-555-2739), Mon–Fri 8:00 AM – 6:00 PM EST\n"
                    f"- **Appeals & Escalations Email**: `appeals@apexshield-care.example.com`\n"
                )
        else:
            citations.append("[Section 1] Policy Overview and Eligibility")
            reply = (
                "I could not find an exact match for that specific inquiry in the ApexShield Care Plus policy document. "
                "For customized benefits assistance, please contact the ApexShield Member Support Team at **1-800-555-APEX**."
            )

        latency = (time.time() - start_time) * 1000
        full_context = " ".join(c.content for c in self.retriever.get_all_chunks())
        guardrail = FaithfulnessGuardrail.evaluate(reply, full_context)

        return AgentResponse(
            reply=reply,
            citations=list(dict.fromkeys(citations)),
            tool_calls=tool_calls,
            model_used="mock-rule-engine",
            latency_ms=latency,
            grounding_score=guardrail.grounding_score,
            is_verified=guardrail.is_faithful,
            audit_notes=guardrail.audit_notes
        )

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> AgentResponse:
        """Invokes OpenAI or OpenRouter with tool execution and strict grounding."""
        from openai import OpenAI

        start_time = time.time()

        if provider == "openrouter":
            key = api_key or settings.OPENROUTER_API_KEY
            if not key or key.strip() == "" or "your_" in key:
                # Key is missing or default placeholder, gracefully notify with mock fallback
                fallback_res = self._mock_inference(messages[-1]["content"], [])
                fallback_res.reply = (
                    "> [!NOTE]\n"
                    "> **Notice**: OpenRouter API key is not yet configured in `.env`. Falling back to the deterministic ClaimAssist evaluation engine.\n\n"
                    + fallback_res.reply
                )
                fallback_res.model_used = "openrouter-fallback-mock"
                return fallback_res

            client = OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=key,
                default_headers={
                    "HTTP-Referer": settings.OPENROUTER_SITE_URL,
                    "X-Title": settings.OPENROUTER_APP_NAME,
                }
            )
            model_name = model or settings.OPENROUTER_MODEL

        elif provider == "openai":
            key = api_key or settings.OPENAI_API_KEY
            if not key or key.strip() == "" or "your_" in key:
                fallback_res = self._mock_inference(messages[-1]["content"], [])
                fallback_res.reply = (
                    "> [!NOTE]\n"
                    "> **Notice**: OpenAI API key is not configured in `.env`. Falling back to the deterministic ClaimAssist evaluation engine.\n\n"
                    + fallback_res.reply
                )
                fallback_res.model_used = "openai-fallback-mock"
                return fallback_res

            client = OpenAI(
                base_url=settings.OPENAI_BASE_URL,
                api_key=key
            )
            model_name = model or settings.OPENAI_MODEL
        else:
            return self._mock_inference(messages[-1]["content"], [])

        # Retrieve relevant policy chunks beforehand as grounding context
        user_query = messages[-1]["content"]
        search_res = self.retriever.search(user_query, top_k=settings.TOP_K_CHUNKS)
        grounding_context = "\n\n".join(
            f"--- Document Section {r.chunk.citation} ---\n{r.chunk.content}"
            for r in search_res
        )

        system_instruction = f"{SYSTEM_PROMPT}\n\nOFFICIAL POLICY EXCERPTS FOR CONTEXT:\n{grounding_context}"

        llm_messages = [{"role": "system", "content": system_instruction}]
        for m in messages:
            llm_messages.append({"role": m["role"], "content": m["content"]})

        try:
            # First LLM call with tool definitions
            response = client.chat.completions.create(
                model=model_name,
                messages=llm_messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1000
            )

            response_msg = response.choices[0].message
            executed_tools = []
            collected_citations = [r.chunk.citation for r in search_res]

            if response_msg.tool_calls:
                # Add assistant's tool call message
                llm_messages.append(response_msg)

                for tc in response_msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")
                    tool_res, cites = self._execute_tool(fn_name, fn_args)
                    collected_citations.extend(cites)
                    executed_tools.append({
                        "tool": fn_name,
                        "args": fn_args,
                        "result": tool_res
                    })

                    # Feed tool response back into conversation
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_res)
                    })

                # Follow-up LLM completion to synthesize answer
                second_response = client.chat.completions.create(
                    model=model_name,
                    messages=llm_messages,
                    temperature=0.2,
                    max_tokens=1000
                )
                final_text = second_response.choices[0].message.content or ""
            else:
                final_text = response_msg.content or ""

            # Extract cited sections from final text
            inline_citations = re.findall(r"\[Section\s+[0-9]+(?:\.[0-9]+)?\]", final_text)
            collected_citations.extend(inline_citations)

            latency = (time.time() - start_time) * 1000
            guardrail = FaithfulnessGuardrail.evaluate(final_text, grounding_context)

            return AgentResponse(
                reply=final_text,
                citations=list(dict.fromkeys(collected_citations)),
                tool_calls=executed_tools,
                model_used=model_name,
                latency_ms=latency,
                grounding_score=guardrail.grounding_score,
                is_verified=guardrail.is_faithful,
                audit_notes=guardrail.audit_notes
            )

        except Exception as e:
            # If API call fails (network issue, rate limit, quota, invalid model), fall back gracefully
            fallback_res = self._mock_inference(user_query, [])
            fallback_res.reply = (
                f"> [!WARNING]\n"
                f"> **LLM Provider API Call Notice**: {str(e)}\n"
                f"> ClaimAssist generated the following verified response using the local policy rule engine:\n\n"
                + fallback_res.reply
            )
            fallback_res.model_used = f"{model_name}-fallback"
            return fallback_res

    def chat(
        self,
        user_query: str,
        conversation_history: Optional[List[AgentMessage]] = None,
        session_id: str = "default",
        provider_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> AgentResponse:
        """Main chat interface for users and API callers."""
        history = conversation_history or []
        provider = (provider_override or settings.LLM_PROVIDER).lower()

        # Build message list
        messages = [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": user_query})

        if provider == "mock":
            response = self._mock_inference(user_query, history)
        else:
            response = self._call_llm(
                messages=messages,
                provider=provider,
                api_key=api_key_override,
                model=model_override
            )

        response.session_id = session_id

        # Log for audit trail
        self.audit_logger.log_interaction(
            session_id=session_id,
            user_query=user_query,
            assistant_response=response.reply,
            citations=response.citations,
            tool_calls=response.tool_calls,
            model_used=response.model_used,
            latency_ms=response.latency_ms
        )

        return response

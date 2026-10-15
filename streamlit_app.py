"""ClaimAssist AI — Interactive Web Application & Portal."""

import streamlit as st
import json
import time
from typing import List

from app.config import settings
from app.retrieval import PolicyRetriever
from app.claims_service import ClaimsService
from app.audit import AuditLogger
from app.agent import ClaimAssistAgent, AgentMessage
from app.algorithms.fraud_detector import ClaimAnomalyDetector
from app.algorithms.accumulator import BenefitAccumulator
from app.algorithms.faithfulness import FaithfulnessGuardrail


# Page configuration
st.set_page_config(
    page_title="ClaimAssist AI — ApexShield Care Plus",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .citation-card {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.88rem;
        color: #334155;
    }
    .status-badge-approved {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-review {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-docs {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .grounding-pill-high {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initialize singletons for retriever, claims service, and agent."""
    retriever = PolicyRetriever()
    claims_service = ClaimsService()
    audit_logger = AuditLogger()
    agent = ClaimAssistAgent(retriever=retriever, claims_service=claims_service, audit_logger=audit_logger)
    return retriever, claims_service, audit_logger, agent


retriever, claims_service, audit_logger, agent = get_services()


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **ClaimAssist AI**, your dedicated assistant for **ApexShield Care Plus**.\n\n"
                       "I can help you:\n"
                       "- Look up your claim status & required actions\n"
                       "- Explain coverage rules, exclusions, and waiting periods with direct citations\n"
                       "- Calculate estimated reimbursement amounts\n"
                       "- Connect you with a human specialist or initiate an appeal",
            "citations": ["[Section 1] Policy Overview and Eligibility", "[Section 8.2] Escalation to Human Claims Specialist"],
            "tool_calls": [],
            "grounding_score": 1.0
        }
    ]

if "api_key" not in st.session_state:
    st.session_state.api_key = settings.OPENROUTER_API_KEY or ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = settings.OPENROUTER_MODEL or "anthropic/claude-3.5-sonnet"


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### ⚙️ System Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=["openrouter", "mock", "openai"],
        index=0 if settings.LLM_PROVIDER == "openrouter" else (1 if settings.LLM_PROVIDER == "mock" else 2),
        help="Select OpenRouter (recommended), Mock (offline test engine), or OpenAI."
    )

    if provider == "openrouter":
        st.markdown("**OpenRouter Configuration**")
        model_options = [
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat",
            "google/gemini-2.5-pro",
            "Custom"
        ]
        choice = st.selectbox("Recommended Model", options=model_options, index=0)
        if choice == "Custom":
            selected_model = st.text_input("Enter OpenRouter Model ID", value="anthropic/claude-3.5-sonnet")
        else:
            selected_model = choice

        api_key_input = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.api_key,
            type="password",
            help="Paste your OpenRouter API key. If left blank, the system uses the local policy evaluation engine."
        )
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.success("API Key updated for this session!")

    elif provider == "openai":
        selected_model = st.text_input("OpenAI Model", value="gpt-4o-mini")
        api_key_input = st.text_input("OpenAI API Key", type="password")
        st.session_state.api_key = api_key_input
    else:
        selected_model = "mock-deterministic-engine"
        st.info("⚡ Mock Mode active: Zero-cost, 100% offline, deterministic rule engine.")

    st.divider()

    # HIPAA Security & Member Identity
    st.markdown("### 🛡️ HIPAA Identity & Role")
    identity_options = [
        "Claims Specialist / Staff (Admin)",
        "Jordan Miller (POL-98214)",
        "Samantha Chen (POL-77312)",
        "Marcus Vance (POL-45109)",
        "Elena Gomez (POL-88123)",
        "David Kim (POL-33291)",
        "Priya Patel (POL-61044)",
        "Unauthenticated Guest"
    ]
    user_role = st.selectbox(
        "Active Session Identity",
        options=identity_options,
        index=0,
        help="Simulate role-based access control and HIPAA Protected Health Information (PHI) authorization."
    )
    is_admin = "Admin" in user_role
    active_pol_id = None
    if "(" in user_role and ")" in user_role:
        active_pol_id = user_role.split("(")[1].split(")")[0]

    st.divider()

    # Fast Claim Status Quick-Widget
    st.markdown("### 🔍 Fast Claim Lookup")
    all_claims = claims_service.list_all_claims()
    claim_ids = [c.claim_id for c in all_claims]
    selected_claim_id = st.selectbox("Select a Sample Claim ID", options=["-- Select --"] + claim_ids)

    if selected_claim_id != "-- Select --":
        claim = claims_service.get_claim(selected_claim_id)
        if claim:
            st.markdown(f"**{claim.claim_id}** — {claim.policyholder_name}")
            status_color = {
                "Approved": "green",
                "Under Review": "orange",
                "Additional Documents Required": "orange",
                "Rejected": "red",
                "Submitted": "blue"
            }.get(claim.status, "gray")
            st.markdown(f"Status: **:{status_color}[{claim.status}]**")
            st.caption(f"Amount: ${claim.claim_amount:,.2f} | Approved: ${claim.approved_amount:,.2f}")
            st.caption(f"Note: {claim.notes}")

            if st.button(f"Ask AI about {claim.claim_id}", key="btn_ask_claim"):
                prompt = f"What is the status of my claim {claim.claim_id} and what are the next steps?"
                st.session_state.pending_prompt = prompt

    st.divider()
    st.markdown("### 📞 Specialist Support")
    st.caption("ApexShield Care Support: **1-800-555-APEX**")
    st.caption("Appeals: `appeals@apexshield-care.example.com`")


# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <div class="main-header">
        <h1>ApexShield ClaimAssist AI</h1>
        <p>Official Policy Inquiry, Claims Tracking, and Human Escalation System</p>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    st.metric(
        label="Policy Coverage",
        value="ApexShield Care+",
        delta="Active"
    )


# Main Navigation Tabs
tab_chat, tab_claims, tab_preclaim, tab_policy, tab_accum, tab_audit = st.tabs([
    "💬 AI Assistant",
    "📋 Claims Registry",
    "⚡ Pre-Claim Adjudicator",
    "📖 Policy Explorer",
    "📈 Benefit Accumulator",
    "📜 Audit Log"
])


# ----------------------------------------------------
# TAB 1: AI CHAT ASSISTANT
# ----------------------------------------------------
with tab_chat:
    st.markdown("#### Ask ClaimAssist AI")
    st.caption("Ask questions regarding your policy coverage, check claims, calculate benefits, or request an appeal.")

    # Suggested Prompts
    st.markdown("**Suggested Inquiries:**")
    prompt_cols = st.columns(3)
    with prompt_cols[0]:
        if st.button("⏱️ Waiting period for pre-existing?", use_container_width=True):
            st.session_state.pending_prompt = "What is the waiting period for pre-existing conditions under ApexShield?"
        if st.button("📋 Status of claim CLM1002?", use_container_width=True):
            st.session_state.pending_prompt = "What is the current status and next action for claim CLM1002?"
    with prompt_cols[1]:
        if st.button("🚫 Why was CLM1004 rejected?", use_container_width=True):
            st.session_state.pending_prompt = "Why was claim CLM1004 rejected and how can I escalate or appeal it?"
        if st.button("👃 Is cosmetic surgery covered?", use_container_width=True):
            st.session_state.pending_prompt = "Does ApexShield Care Plus cover cosmetic rhinoplasty surgery?"
    with prompt_cols[2]:
        if st.button("💵 Calculate $4,000 surgery bill", use_container_width=True):
            st.session_state.pending_prompt = "Estimate reimbursement for a $4,000 in-network surgery with $500 deductible remaining."
        if st.button("🧑‍💼 Speak with human specialist", use_container_width=True):
            st.session_state.pending_prompt = "I need to escalate my claim and speak with a human claims specialist."

    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📌 Policy Citations & Sources", expanded=False):
                    for cite in msg["citations"]:
                        st.markdown(f"- **{cite}**")
            if msg.get("tool_calls"):
                with st.expander("🛠️ Actions & Tools Triggered", expanded=False):
                    st.json(msg["tool_calls"])

    # Handle incoming user input
    user_input = st.chat_input("Ask a question about your policy or enter a claim ID (e.g. CLM1001)...")
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("ClaimAssist is analyzing policy terms & claims records..."):
                history = [
                    AgentMessage(role=m["role"], content=m["content"])
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ["user", "assistant"]
                ]

                response = agent.chat(
                    user_query=user_input,
                    conversation_history=history,
                    provider_override=provider,
                    api_key_override=st.session_state.api_key if provider != "mock" else None,
                    model_override=selected_model if provider != "mock" else None
                )

                st.markdown(response.reply)

                # Grounding and Verification Badge
                badge_color = "green" if response.is_verified else "orange"
                st.markdown(f"""
                <div style="margin-top: 10px;">
                    <span class="grounding-pill-high">🛡️ Grounding Confidence: {int(response.grounding_score * 100)}%</span>
                    <span style="font-size: 0.8rem; color: #64748b; margin-left: 10px;">{response.audit_notes}</span>
                </div>
                """, unsafe_allow_html=True)

                if response.citations:
                    with st.expander("📌 Policy Citations & Sources", expanded=True):
                        for cite in response.citations:
                            st.markdown(f"- **{cite}**")

                if response.tool_calls:
                    with st.expander("🛠️ Actions & Tools Triggered", expanded=False):
                        st.json(response.tool_calls)

                st.caption(f"Engine: `{response.model_used}` | Latency: `{response.latency_ms:.1f}ms`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.reply,
            "citations": response.citations,
            "tool_calls": response.tool_calls,
            "grounding_score": response.grounding_score
        })


# ----------------------------------------------------
# TAB 2: CLAIMS REGISTRY
# ----------------------------------------------------
with tab_claims:
    st.markdown("### 📋 ApexShield Claims Database")
    st.caption("Real-time view of submitted member claims, adjudication status, and automated anomaly auditing.")

    col_flt1, col_flt2 = st.columns([2, 1])
    with col_flt1:
        search_kw = st.text_input("Search claims by ID, Member Name, or Category", placeholder="e.g. Marcus Vance or Emergency")
    with col_flt2:
        status_filter = st.selectbox("Filter by Status", options=["All", "Approved", "Under Review", "Additional Documents Required", "Rejected", "Submitted"])

    claims_data = claims_service.list_all_claims()
    if search_kw:
        claims_data = [c for c in claims_data if search_kw.lower() in f"{c.claim_id} {c.policyholder_name} {c.claim_type}".lower()]
    if status_filter != "All":
        claims_data = [c for c in claims_data if c.status == status_filter]

    for c in claims_data:
        status_class = {
            "Approved": "status-badge-approved",
            "Under Review": "status-badge-review",
            "Additional Documents Required": "status-badge-docs",
            "Rejected": "status-badge-rejected",
            "Submitted": "status-badge-review"
        }.get(c.status, "status-badge-review")

        # HIPAA Verification Check
        is_auth, auth_msg = claims_service.verify_claim_access(
            c.claim_id,
            requester_id=active_pol_id,
            is_admin=is_admin
        )

        if not is_auth:
            with st.container():
                st.markdown(f"""
                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #9f1239;">🔒 {c.claim_id} &bull; Protected Health Information (PHI)</span>
                        <span class="status-badge-rejected">HIPAA Restricted</span>
                    </div>
                    <div style="color: #881337; font-size: 0.85rem; margin-top: 4px;">
                        {auth_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            continue

        with st.container():
            st.markdown(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{c.claim_id} &bull; {c.policyholder_name}</span>
                    <span class="{status_class}">{c.status}</span>
                </div>
                <div style="color: #64748b; font-size: 0.88rem; margin-top: 4px;">
                    Policyholder ID: <code>{c.policyholder_id}</code> | Type: <strong>{c.claim_type}</strong> | Submitted: {c.submission_date}
                </div>
                <div style="margin-top: 10px; font-size: 0.92rem;">
                    <strong>Claim Amount:</strong> ${c.claim_amount:,.2f} &nbsp;|&nbsp; <strong>Approved:</strong> ${c.approved_amount:,.2f}
                </div>
                <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; margin-top: 8px; font-size: 0.85rem; color: #334155;">
                    <strong>Next Action:</strong> {c.next_action}<br>
                    <strong>Adjudication Notes:</strong> {c.notes}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(f"🔍 Run Anomaly Audit for {c.claim_id}", key=f"btn_anom_{c.claim_id}"):
                    all_member_claims = [
                        claim_item.model_dump() for claim_item in claims_service.list_all_claims()
                        if claim_item.policyholder_id.upper() == c.policyholder_id.upper()
                    ]
                    report = ClaimAnomalyDetector.evaluate_claim(
                        claim_id=c.claim_id,
                        claim_type=c.claim_type,
                        claim_amount=c.claim_amount,
                        submission_date=c.submission_date,
                        prior_claims=all_member_claims
                    )
                    risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(report.risk_level, "gray")
                    st.markdown(f"**Risk Level**: **:{risk_color}[{report.risk_level}]** (Score: {report.risk_score:.2f}, Z-Score: {report.z_score})")
                    if report.is_potential_duplicate:
                        st.error("🚨 Potential Duplicate Claim Detected!")
                    st.caption(f"Recommendation: {report.recommendation}")
                    if report.flags:
                        for f in report.flags:
                            st.warning(f)
                    else:
                        st.success("No anomalies detected. Claim matches standard baseline.")

            with col_btn2:
                if st.button(f"📝 Generate Formal Appeal Letter ({c.claim_id})", key=f"btn_app_{c.claim_id}"):
                    letter = claims_service.generate_appeal_letter(c.claim_id)
                    st.download_button(
                        label=f"⬇️ Download Appeal Packet ({c.claim_id})",
                        data=letter["letter_content"],
                        file_name=f"Appeal_Packet_{c.claim_id}.txt",
                        mime="text/plain"
                    )
                    with st.expander("Preview Form CL-01 Appeal Notice", expanded=True):
                        st.text(letter["letter_content"])


# ----------------------------------------------------
# TAB 3: PRE-CLAIM ADJUDICATOR
# ----------------------------------------------------
with tab_preclaim:
    st.markdown("### ⚡ Pre-Claim Adjudication Simulator")
    st.caption("Verify whether a medical procedure will be covered before receiving treatment or submitting a bill.")

    col_pc1, col_pc2 = st.columns(2)
    with col_pc1:
        tx_name = st.selectbox(
            "Proposed Treatment / Procedure",
            options=[
                "Emergency Room Trauma Care",
                "Diagnostic MRI Scan ($1,200)",
                "Pre-Existing Asthma Treatment",
                "Maternity & Normal Delivery",
                "Elective Cataract Surgery",
                "Cosmetic Rhinoplasty",
                "Routine Dental Cleaning",
                "Custom Procedure Name"
            ]
        )
        if tx_name == "Custom Procedure Name":
            tx_name = st.text_input("Enter Treatment Name", value="Inpatient Knee Surgery")

        tx_cost = st.number_input("Estimated Total Cost ($)", min_value=50.0, value=2500.0, step=100.0)

    with col_pc2:
        enrolled_mos = st.slider("Continuous Coverage Duration (Months)", min_value=0, max_value=24, value=6)
        is_pre_ex = st.checkbox("Condition diagnosed within 24 months before enrollment?", value=False)
        in_network_tier = st.checkbox("In-Network Provider Network", value=True)

    if st.button("🚀 Run Pre-Claim Adjudication Simulation", type="primary"):
        res_sim = claims_service.adjudicate_pre_claim(
            treatment_name=tx_name,
            estimated_cost=tx_cost,
            months_enrolled=enrolled_mos,
            is_pre_existing=is_pre_ex,
            in_network=in_network_tier
        )

        dec_color = {"Covered": "green", "Waiting Period Active": "orange", "Excluded": "red"}.get(res_sim["decision"], "gray")
        st.markdown(f"### Adjudication Decision: **:{dec_color}[{res_sim['decision']}]**")
        st.markdown(f"**Explanation**: {res_sim['notes']}")

        col_sr1, col_sr2 = st.columns(2)
        with col_sr1:
            st.metric("Estimated Plan Reimbursement", f"${res_sim['estimated_reimbursement']:,.2f}")
        with col_sr2:
            st.metric("Estimated Member Out-of-Pocket", f"${res_sim['patient_estimated_share']:,.2f}")

        st.markdown("**Governing Policy Sections:**")
        for g in res_sim["governing_sections"]:
            st.markdown(f"- `{g}`")

        st.markdown("**Mandatory Documentation Required per Section 6:**")
        for doc in res_sim["required_documents"]:
            st.markdown(f"- 📄 {doc}")


# ----------------------------------------------------
# TAB 4: POLICY EXPLORER
# ----------------------------------------------------
with tab_policy:
    st.markdown("### 📖 ApexShield Care Plus — Policy Knowledge Base")
    st.caption("Hybrid BM25Okapi & Semantic Vector Retrieval with Exact Citations.")

    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        policy_query = st.text_input("Semantic Policy Search", placeholder="e.g. cataract surgery waiting period or out-of-pocket maximum")
    with col_p2:
        top_k_select = st.slider("Top Results", min_value=1, max_value=6, value=3)

    if policy_query:
        search_results = retriever.search(policy_query, top_k=top_k_select)
        st.markdown(f"**Found {len(search_results)} relevant policy sections (Hybrid RRF):**")
        for res in search_results:
            st.markdown(f"""
            <div class="citation-card">
                <span style="font-weight: 700; color: #0369a1; font-size: 1rem;">{res.chunk.citation}</span>
                <span style="float: right; color: #64748b; font-size: 0.8rem;">BM25: {res.bm25_score:.2f} | Dense: {res.semantic_score:.2f}</span>
                <p style="margin-top: 6px; margin-bottom: 0;">{res.chunk.content}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        all_chunks = retriever.get_all_chunks()
        categories = sorted(list(set(c.category for c in all_chunks)))
        selected_category = st.selectbox("Filter by Policy Chapter", options=["All"] + categories)

        for c in all_chunks:
            if selected_category == "All" or c.category == selected_category:
                with st.expander(f"{c.citation} ({c.category})"):
                    st.markdown(c.content)


# ----------------------------------------------------
# TAB 5: BENEFIT ACCUMULATOR
# ----------------------------------------------------
with tab_accum:
    st.markdown("### 📈 Deductible & Out-of-Pocket Maximum Accumulator")
    st.caption("State machine modeling Section 5.1 ($1,000 Deductible) and Section 5.2 ($4,500 Out-of-Pocket Max).")

    pol_id_input = st.text_input("Policyholder ID", value="POL-98214")
    bill_input = st.number_input("Medical Expense Amount ($)", min_value=100.0, value=3200.0, step=100.0)
    in_net_accum = st.checkbox("Treatment in Participating Network (80%)", value=True)

    if st.button("Process Claim & Accrue Benefits", type="primary"):
        accum_res = agent.accumulator.adjudicate_and_accumulate(pol_id_input, bill_input, in_net_accum)

        st.success("Claim processed through accumulator state machine!")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Deductible Applied", f"${accum_res['deductible_applied']:,.2f}")
            st.caption(f"Remaining Deductible: ${accum_res['deductible_remaining_after']:,.2f}")
        with col_a2:
            st.metric("ApexShield Payout", f"${accum_res['plan_payout']:,.2f}")
            if accum_res["catastrophic_100_percent_active"]:
                st.info("🎉 100% Catastrophic Coverage Active (OOP Max Reached)!")
        with col_a3:
            st.metric("Patient Total OOP Accrued", f"${accum_res['oop_accumulated_total']:,.2f}")
            st.caption(f"Remaining OOP Cap: ${accum_res['oop_max_remaining']:,.2f}")


# ----------------------------------------------------
# TAB 6: AUDIT LOG
# ----------------------------------------------------
with tab_audit:
    st.markdown("### 📜 Conversation & Compliance Audit Log")
    st.caption("Logged conversation turns with timestamps, citations, and latency metrics stored in `logs/conversations.jsonl`.")

    if st.button("🔄 Refresh Audit Logs"):
        st.rerun()

    recent_logs = audit_logger.read_recent_logs(limit=25)
    if not recent_logs:
        st.info("No conversation logs recorded yet. Send a query in the AI Assistant tab to see records here.")
    else:
        for entry in recent_logs:
            with st.expander(f"{entry['timestamp']} — {entry['user_query'][:60]}..."):
                st.markdown(f"**Query**: {entry['user_query']}")
                st.markdown(f"**Assistant Response**:\n{entry['assistant_response']}")
                st.markdown(f"**Citations**: {', '.join(entry['citations']) if entry['citations'] else 'None'}")
                st.markdown(f"**Model**: `{entry['model_used']}` | **Latency**: `{entry['latency_ms']}ms`")
                if entry.get("tool_calls"):
                    st.json(entry["tool_calls"])

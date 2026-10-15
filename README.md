# 🛡️ ClaimAssist AI — Intelligent Health Insurance Claim Adjudication & Support

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.2-61dafb.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.3-646cff.svg)](https://vitejs.dev)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Claude%203.5%20Sonnet-purple.svg)](https://openrouter.ai)
[![SQLite](https://img.shields.io/badge/Storage-SQLite%20Persistent-blue.svg)](https://sqlite.org)
[![Tests Passing](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)]()

**ClaimAssist AI** is an enterprise-grade AI conversational platform for healthcare insurance adjudication, policy explanation, claims status tracking, benefit calculations, and human specialist escalation. Built specifically around the **ApexShield Care Plus** policy.

---

## 🌟 Key Features

1. **Strictly Grounded Policy RAG**:
   - Parses the complete **ApexShield Care Plus** policy into 27 structured, cited chunks.
   - **BM25Okapi** lexical retrieval combined with medical synonym expansion and **Reciprocal Rank Fusion (RRF)**.
   - Every answer guarantees verified citations (e.g., `[Section 2.1]`, `[Section 3.2]`, `[Section 4.1]`, `[Section 8.2]`).
   - Zero hallucination policy: explicitly communicates exclusions and waiting periods.

2. **Real-time Claim Adjudication & Tracking**:
   - Direct integration with persistent SQLite customer claims registry (`data/claimassist.db`).
   - Look up claims (`CLM1001` - `CLM1006`) by Claim ID or Policyholder.
   - Real-time status reporting: *Approved*, *Under Review*, *Additional Documents Required*, *Rejected*, *Submitted*.

3. **HIPAA Security & Access Control**:
   - Role-Based Access Control (RBAC) preventing unauthorized access to Protected Health Information (PHI).
   - Only authorized policyholders or Claims Staff/SIU can access claim notes and financial disbursements.

4. **Human Specialist Escalation Engine & Form CL-01 Appeals**:
   - Compliant with **Section 8.1, 8.2 & 8.3** of the policy.
   - Automatically detects disputed claim denials or review delays exceeding 20 days.
   - Generates pre-populated Form CL-01 appeal notices and unique tracking tickets routed to the *Human Adjudication & Appeals Unit*.

5. **Financial Benefit Accumulator**:
   - Stateful calculation of annual deductibles ($1,000 Ind / $2,500 Fam) and Out-of-Pocket Maximums ($4,500 Ind / $9,000 Fam).
   - Automatically triggers 100% plan payout when catastrophic thresholds are met per Section 5.2.

6. **Algorithmic Integrity Guardrails**:
   - **Actuarial Z-Score Anomaly Detector**: Flags unusual billing spikes.
   - **Timely Filing Validator**: Checks Section 5.4 90-day filing limit.
   - **Duplicate Claim Verification**: Detects overlapping dates and identical treatments.
   - **Faithfulness Guardrail**: Verifies cited section legitimacy and numerical fact entailment.

---

## 🏛️ High-Level Design (HLD)

```mermaid
flowchart TD
    subgraph ClientLayer ["1. Presentation Layer"]
        ReactApp["React 19 + Vite SPA (Port 5173)"]
        StreamlitApp["Streamlit Portal (Port 8501)"]
        CurlClient["External REST API Client (EHR / Payers)"]
    end

    subgraph GatewayLayer ["2. API & Security Gateway (FastAPI :8000)"]
        FastAPIApp["FastAPI REST Router"]
        HIPAAGuard["HIPAA Authorization Guard (RBAC)"]
        CORSMiddleware["CORS & Pydantic v2 Validation"]
    end

    subgraph AgentLayer ["3. Intelligent Agent Core (ClaimAssistAgent)"]
        InferenceRouter{"Inference Router"}
        OpenRouterLLM["OpenRouter API Client (Claude 3.5 Sonnet)"]
        MockEngine["Deterministic Mock Engine (Offline Fallback)"]
        ContextMemory["Multi-Turn Context Memory"]
        ToolDispatcher["Function Calling & Tool Dispatcher"]
    end

    subgraph AlgoLayer ["4. Domain Algorithms & Integrity Guardrails"]
        BM25["BM25Okapi + Medical Synonym Expansion + RRF"]
        Accumulator["Benefit Accumulator State Machine (Section 5.1 & 5.2)"]
        FraudSIU["SIU Anomaly & Duplicate Claim Detector (Z-Score + 90-Day Filing)"]
        Guardrail["Faithfulness & Numerical Entailment Guardrail"]
        Adjudicator["Pre-Claim Adjudicator & Form CL-01 Appeal Packet"]
    end

    subgraph DataLayer ["5. Persistence & Storage Layer"]
        SQLiteDB[("SQLite 3 Database (claimassist.db)")]
        PolicyMD[("ApexShield Policy Markdown (27 Clauses)")]
        AuditLog[("Structured Audit Trails (conversations.jsonl)")]
    end

    ClientLayer --> CORSMiddleware
    CORSMiddleware --> HIPAAGuard
    HIPAAGuard --> FastAPIApp

    FastAPIApp --> InferenceRouter
    InferenceRouter -->|Online LLM| OpenRouterLLM
    InferenceRouter -->|Offline / Fallback| MockEngine

    MockEngine <--> ContextMemory
    OpenRouterLLM --> ToolDispatcher
    MockEngine --> ToolDispatcher

    ToolDispatcher --> BM25
    ToolDispatcher --> Accumulator
    ToolDispatcher --> FraudSIU
    ToolDispatcher --> Adjudicator

    OpenRouterLLM --> Guardrail
    MockEngine --> Guardrail
    Guardrail --> FastAPIApp

    BM25 <--> PolicyMD
    Accumulator <--> SQLiteDB
    Adjudicator <--> SQLiteDB
    FraudSIU <--> SQLiteDB
    FastAPIApp --> AuditLog
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/ayush300302/claim_assist_ai.git
cd claim_assist_ai

# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment & OpenRouter API Key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and paste your OpenRouter API key:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

> **Note**: If `OPENROUTER_API_KEY` is not provided, ClaimAssist automatically runs in deterministic **Mock Mode**, ensuring full offline functionality and zero failures.

---

## 💻 Running the Application

### Option A: Launch the Streamlit Web Application

```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch the FastAPI Backend Server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation available at: `http://127.0.0.1:8000/docs`.

---

## 🧪 Running Automated Tests

Run the full pytest suite:

```bash
pytest tests/ -v
```

All 16 tests cover:
- Policy chunking & TF-IDF hybrid retrieval
- Claims lookup, search, and validation
- Reimbursement calculations under Section 5.3
- Human specialist escalation ticket creation
- Agent grounding and citation extraction
- All FastAPI REST API endpoints

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send message and receive grounded response with citations & tool calls |
| `GET` | `/api/claims` | List claims or filter by keyword |
| `GET` | `/api/claims/{claim_id}` | Retrieve specific claim record |
| `POST` | `/api/claims/{claim_id}/escalate` | Escalate a claim dispute or delay to a human specialist |
| `POST` | `/api/reimbursement/calculate` | Calculate co-insurance and deductible breakdown |
| `POST` | `/api/policy/search` | Semantic and keyword search across policy chunks |
| `GET` | `/api/policy/sections` | List all indexed policy sections and citations |
| `GET` | `/health` | System health check and index metrics |

---

## 📄 License

This fictional project is licensed under the MIT License.

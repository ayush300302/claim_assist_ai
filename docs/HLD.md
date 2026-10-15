# ClaimAssist AI — High-Level Design (HLD) Specification

## 1. Executive Architecture Overview

**ClaimAssist AI** is an enterprise-grade autonomous claims adjudication, policy retrieval, and appeals platform designed for health insurers (modeled around the comprehensive 8-chapter **ApexShield Care Plus** policy).

---

## 2. High-Level Architecture Diagram (Mermaid)

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

## 3. Layer Specifications

### 3.1 Presentation Layer
- **React 19 + Vite SPA (`frontend/`)**: Modern fintech glassmorphic portal, tabbed navigation, real-time grounding audit badges, suggested prompts, and role switcher.
- **Streamlit (`streamlit_app.py`)**: Alternative analytical workspace with real-time claims and audit log reader.

### 3.2 Security & HIPAA Verification Layer
- Role-based authorization: Validates whether requester ID matches claim policyholder ID.
- Blocks unpermitted users from reading claim notes, diagnosis information, and payment disbursement accounts.

### 3.3 Agent Orchestration Layer
- **Multi-Turn Context**: Resolves pronouns and contextual follow-ups ("appeal it", "what's the status?") by scanning conversation history for prior claim entities.
- **Hybrid Inference**: Supports native OpenRouter function calling with sub-50ms offline deterministic mock engine.

### 3.4 Domain Algorithms
1. **Hybrid Retrieval**: BM25Okapi + Domain Synonym Expansion + Reciprocal Rank Fusion (RRF).
2. **Benefit Accumulator**: Stateful calculation of Deductible and Out-of-Pocket Maximum per Sections 5.1 & 5.2.
3. **SIU Anomaly Detector**: Actuarial Z-Score, 90-day timely filing check, and duplicate claim detection.
4. **Faithfulness Guardrail**: Verification of cited section legality and strict numerical fact entailment.

### 3.5 Persistence Layer
- **SQLite 3 (`data/claimassist.db`)**: Thread-safe persistent tables for `claims`, `accumulators`, and `escalations`.

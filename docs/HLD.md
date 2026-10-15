# ClaimAssist AI — High-Level Design (HLD) Specification

## 1. Executive Architecture Overview

**ClaimAssist AI** is an enterprise-grade autonomous claims adjudication, policy retrieval, and appeals platform designed for health insurers (modeled around the comprehensive 8-chapter **ApexShield Care Plus** policy).

---

## 2. High-Level Architecture Diagram (Mermaid)

```mermaid
flowchart TB
    subgraph ClientLayer ["1. Presentation Layer"]
        ReactApp["React 19 + Vite SPA<br/>(Port 5173 - Glassmorphic UI)"]
        StreamlitApp["Streamlit Multi-Tab Portal<br/>(Port 8501 - Internal Tool)"]
        CurlClient["External REST API Client<br/>(Partner Payers / EHR / Postman)"]
    end

    subgraph GatewayLayer ["2. API & Security Gateway (FastAPI :8000)"]
        FastAPIApp["FastAPI REST Router<br/>(/api/chat, /api/claims, /api/accumulator)"]
        HIPAAGuard["HIPAA Authorization Guard<br/>(Ownership & Role-Based Access Control)"]
        CORSMiddleware["CORS & Request Validation<br/>(Pydantic v2 Models)"]
    end

    subgraph AgentLayer ["3. Intelligent Agent Core (ClaimAssistAgent)"]
        InferenceRouter{"Inference Router<br/>(Provider Flag)"}
        OpenRouterLLM["OpenRouter API Client<br/>(Claude 3.5 Sonnet / Llama 3.3 / GPT-4o)"]
        MockEngine["Deterministic Mock Engine<br/>(Sub-50ms Offline Fallback)"]
        ContextMemory["Multi-Turn Context Resolver<br/>(Active Claim & Policyholder State)"]
        ToolDispatcher["Function Calling & Tool Dispatcher"]
    end

    subgraph AlgoLayer ["4. Domain Algorithms & Integrity Guardrails"]
        direction TB
        
        subgraph RAGModule ["Hybrid Policy Retrieval"]
            BM25["BM25Okapi Search<br/>(Term Saturation k1=1.5, b=0.75)"]
            Synonyms["Medical Synonym Expansion<br/>(Clinical & Colloquial Mappings)"]
            RRF["Reciprocal Rank Fusion<br/>(RRF k=60 Scoring)"]
        end

        subgraph AccumulatorModule ["Financial Benefit Accumulator"]
            StateEngine["Finite-State Machine<br/>(Section 5.1 Deductible & 5.2 OOP Max)"]
            CatastrophicTrigger["100 Percent Plan Payout Trigger<br/>(Catastrophic Cap Protection)"]
        end

        subgraph FraudModule ["SIU Fraud & Anomaly Engine"]
            ZScore["Z-Score Outlier Detector<br/>(Actuarial Distribution Benchmarks)"]
            FilingCheck["Timely Filing Verifier<br/>(Section 5.4: 90-Day Deadline)"]
            DupCheck["Duplicate Claim Detector<br/>(Overlapping Dates & Same Treatment)"]
        end

        subgraph GuardrailModule ["Faithfulness Guardrail"]
            CitationValidator["Citation Authenticator<br/>(Known Section Whitelist)"]
            NumEntailment["Numerical Fact Entailment<br/>(Dollar, Percentage & Time Windows)"]
        end

        subgraph AdjudicationModule ["Claims & Legal Services"]
            PreClaimAdj["Pre-Claim Adjudicator<br/>(Sections 3 & 4 Waiting/Exclusions)"]
            AppealGen["Form CL-01 Appeal Generator<br/>(Section 8.1 Formal Legal Packet)"]
        end
    end

    subgraph DataLayer ["5. Persistence & Storage Layer"]
        SQLiteDB[("SQLite 3 Database<br/>(claimassist.db)")]
        ClaimsTable["Claims Registry Table"]
        AccumulatorsTable["Accumulators State Table"]
        EscalationsTable["Escalation Tickets Table"]
        PolicyMD[("ApexShield Care Plus Policy<br/>(27 Granular Clauses Markdown)")]
        AuditLog[("Structured Audit Trails<br/>(logs/conversations.jsonl)")]
    end

    ClientLayer --> CORSMiddleware
    CORSMiddleware --> HIPAAGuard
    HIPAAGuard --> FastAPIApp

    FastAPIApp --> InferenceRouter
    InferenceRouter -->|Online Mode| OpenRouterLLM
    InferenceRouter -->|Offline / Fallback| MockEngine

    MockEngine <--> ContextMemory
    OpenRouterLLM --> ToolDispatcher
    MockEngine --> ToolDispatcher

    ToolDispatcher --> RAGModule
    ToolDispatcher --> AccumulatorModule
    ToolDispatcher --> FraudModule
    ToolDispatcher --> AdjudicationModule

    OpenRouterLLM --> GuardrailModule
    MockEngine --> GuardrailModule
    GuardrailModule --> FastAPIApp

    Synonyms --> BM25
    BM25 --> RRF
    RRF <--> PolicyMD

    StateEngine --> CatastrophicTrigger
    StateEngine <--> AccumulatorsTable
    AdjudicationModule <--> ClaimsTable
    AdjudicationModule --> EscalationsTable

    ClaimsTable --- SQLiteDB
    AccumulatorsTable --- SQLiteDB
    EscalationsTable --- SQLiteDB

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

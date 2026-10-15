import React, { useState, useEffect, useRef } from 'react';
import { 
  Shield, 
  MessageSquare, 
  FileText, 
  Calculator, 
  Sparkles, 
  Search, 
  AlertTriangle, 
  CheckCircle2, 
  Send, 
  RefreshCw, 
  Lock, 
  Unlock, 
  Copy, 
  Download, 
  ExternalLink,
  ChevronRight,
  TrendingUp,
  Activity,
  Layers,
  HelpCircle,
  X
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

const IDENTITIES = [
  { label: 'Jordan Miller (POL-98214)', id: 'POL-98214', name: 'Jordan Miller', role: 'member', token: 'token-pol-98214' },
  { label: 'Samantha Chen (POL-77312)', id: 'POL-77312', name: 'Samantha Chen', role: 'member', token: 'token-pol-77312' },
  { label: 'Marcus Vance (POL-45109)', id: 'POL-45109', name: 'Marcus Vance', role: 'member', token: 'token-pol-45109' },
  { label: 'Elena Gomez (POL-88123)', id: 'POL-88123', name: 'Elena Gomez', role: 'member', token: 'token-pol-88123' },
  { label: 'Claims Specialist / SIU (Admin)', id: 'ADMIN-STAFF', name: 'Specialist Miller', role: 'admin', token: 'admin-secret-token' },
  { label: 'Anonymous Guest', id: null, name: 'Guest', role: 'guest', token: null }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'claims' | 'preclaim' | 'accumulator' | 'policy'
  const [currentIdentity, setCurrentIdentity] = useState(IDENTITIES[0]);
  const [provider, setProvider] = useState('openrouter'); // 'openrouter' | 'mock'
  const [apiKey, setApiKey] = useState('');
  const [serverHealth, setServerHealth] = useState(null);

  // Chat State
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Hello! I am **ClaimAssist AI**, your intelligent insurance copilot for **ApexShield Care Plus**.\n\nI can verify policy clauses per verified chapters, calculate Section 5.3 reimbursements, audit claims for anomalies, or file formal Form CL-01 appeals.\n\nHow can I help you today?`,
      citations: ['[Section 1] Policy Overview and Eligibility'],
      grounding_score: 1.0,
      tool_calls: []
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const messagesEndRef = useRef(null);

  // Claims State
  const [claims, setClaims] = useState([]);
  const [claimsSearch, setClaimsSearch] = useState('');
  const [selectedClaimModal, setSelectedClaimModal] = useState(null);
  const [appealModal, setAppealModal] = useState(null);
  const [anomalyModal, setAnomalyModal] = useState(null);

  // Pre-claim Simulator State
  const [treatmentName, setTreatmentName] = useState('Emergency Appendectomy');
  const [estimatedCost, setEstimatedCost] = useState(3200);
  const [monthsEnrolled, setMonthsEnrolled] = useState(12);
  const [isPreExisting, setIsPreExisting] = useState(false);
  const [preclaimResult, setPreclaimResult] = useState(null);
  const [isLoadingPreclaim, setIsLoadingPreclaim] = useState(false);

  // Accumulator State
  const [accumulatorData, setAccumulatorData] = useState(null);
  const [simulatedCharge, setSimulatedCharge] = useState(1200);

  // Policy Search State
  const [policyQuery, setPolicyQuery] = useState('');
  const [policyResults, setPolicyResults] = useState([]);
  const [allSections, setAllSections] = useState([]);

  // Check Backend Health on Mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => setServerHealth(data))
      .catch(() => setServerHealth({ status: 'offline' }));
    
    loadClaims();
    loadAllPolicySections();
    loadAccumulator();
  }, [currentIdentity]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const getAuthHeaders = () => {
    const headers = { 'Content-Type': 'application/json' };
    if (currentIdentity && currentIdentity.token) {
      headers['Authorization'] = `Bearer ${currentIdentity.token}`;
    }
    return headers;
  };

  const loadClaims = () => {
    if (!currentIdentity || !currentIdentity.token) {
      setClaims([]);
      return;
    }
    fetch(`${API_BASE}/api/claims`, {
      headers: getAuthHeaders()
    })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (Array.isArray(data)) {
          setClaims(data);
        } else {
          setClaims([]);
        }
      })
      .catch(err => {
        console.error("Failed to load claims", err);
        setClaims([]);
      });
  };

  const loadAllPolicySections = () => {
    fetch(`${API_BASE}/api/policy/sections`)
      .then(res => res.json())
      .then(data => setAllSections(data))
      .catch(err => console.error("Failed to load policy sections", err));
  };

  const loadAccumulator = () => {
    const polId = currentIdentity.id || 'POL-98214';
    fetch(`${API_BASE}/api/accumulator/adjudicate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ policyholder_id: polId, eligible_charges: 0 })
    })
      .then(res => res.json())
      .then(data => setAccumulatorData(data))
      .catch(err => console.error(err));
  };

  // Chat Submission
  const handleSendMessage = async (textToSend) => {
    const query = textToSend || inputMessage;
    if (!query.trim() || isLoadingChat) return;

    const newHistory = [...messages, { role: 'user', content: query }];
    setMessages(newHistory);
    setInputMessage('');
    setIsLoadingChat(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          message: query,
          history: newHistory.filter(m => m.role !== 'system').map(m => ({ role: m.role, content: m.content })),
          provider: provider,
          api_key: apiKey || null,
          session_id: `session-${currentIdentity.id || 'guest'}`
        })
      });

      const data = await response.json();
      setMessages([...newHistory, {
        role: 'assistant',
        content: data.reply,
        citations: data.citations || [],
        tool_calls: data.tool_calls || [],
        grounding_score: data.grounding_score ?? 0.95,
        model_used: data.model_used,
        latency_ms: data.latency_ms
      }]);
    } catch (err) {
      setMessages([...newHistory, {
        role: 'assistant',
        content: `> [!WARNING]\n> Could not connect to API backend at \`${API_BASE}\`. Please ensure the backend is running.`,
        citations: [],
        grounding_score: 0.0
      }]);
    } finally {
      setIsLoadingChat(false);
    }
  };

  // Run Pre-Claim Adjudication
  const handlePreclaimAdjudicate = async () => {
    setIsLoadingPreclaim(true);
    try {
      const res = await fetch(`${API_BASE}/api/claims/adjudicate-pre-claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          treatment_name: treatmentName,
          estimated_cost: parseFloat(estimatedCost),
          months_enrolled: parseInt(monthsEnrolled),
          is_pre_existing: isPreExisting,
          in_network: true
        })
      });
      const data = await res.json();
      setPreclaimResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoadingPreclaim(false);
    }
  };

  // Accumulate Expense
  const handleAccumulateExpense = async () => {
    try {
      const polId = currentIdentity.id || 'POL-98214';
      const res = await fetch(`${API_BASE}/api/accumulator/adjudicate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          policyholder_id: polId,
          eligible_charges: parseFloat(simulatedCharge),
          in_network: true
        })
      });
      const data = await res.json();
      setAccumulatorData(data);
    } catch (err) {
      console.error(err);
    }
  };

  // Run Policy Search
  const handlePolicySearch = async (e) => {
    e?.preventDefault();
    if (!policyQuery.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/api/policy/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: policyQuery, top_k: 4 })
      });
      const data = await res.json();
      setPolicyResults(data);
    } catch (err) {
      console.error(err);
    }
  };

  // Anomaly Audit Check
  const runAnomalyAudit = async (claimId) => {
    try {
      const res = await fetch(`${API_BASE}/api/claims/${claimId}/anomaly`, {
        headers: getAuthHeaders()
      });
      const data = await res.json();
      setAnomalyModal(data);
    } catch (err) {
      console.error(err);
    }
  };

  // Generate Appeal Letter
  const runGenerateAppeal = async (claimId) => {
    try {
      const res = await fetch(`${API_BASE}/api/claims/${claimId}/appeal-letter`, {
        method: 'POST',
        headers: getAuthHeaders()
      });
      const data = await res.json();
      setAppealModal(data);
    } catch (err) {
      console.error(err);
    }
  };

  // Check HIPAA permission for claim
  const isClaimAuthorized = (claim) => {
    if (currentIdentity.role === 'admin') return true;
    if (!currentIdentity.id) return false;
    const cleanReq = currentIdentity.id.replace(/[^A-Za-z0-9]/g, '').toUpperCase();
    const cleanPol = claim.policyholder_id.replace(/[^A-Za-z0-9]/g, '').toUpperCase();
    return cleanReq === cleanPol;
  };

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <Shield size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-name">ClaimAssist AI</span>
              <span className="brand-badge">ApexShield Care</span>
            </div>
          </div>
        </div>

        <div className="header-controls">
          {/* HIPAA User Switcher */}
          <div className="hipaa-pill">
            {currentIdentity.role === 'admin' ? (
              <Unlock size={16} color="#10b981" />
            ) : currentIdentity.id ? (
              <Shield size={16} color="#3b82f6" />
            ) : (
              <Lock size={16} color="#ef4444" />
            )}
            <span style={{ fontSize: '0.78rem' }}>Session Identity:</span>
            <select 
              className="hipaa-select"
              value={currentIdentity.id || 'guest'}
              onChange={(e) => {
                const found = IDENTITIES.find(i => (i.id || 'guest') === e.target.value);
                setCurrentIdentity(found);
              }}
            >
              {IDENTITIES.map((id, idx) => (
                <option key={idx} value={id.id || 'guest'}>
                  {id.label}
                </option>
              ))}
            </select>
          </div>

          {/* Provider Selector */}
          <div className="hipaa-pill">
            <Sparkles size={16} color="#10b981" />
            <select 
              className="hipaa-select"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              <option value="openrouter">OpenRouter (Claude 3.5)</option>
              <option value="mock">Offline Mock Engine</option>
            </select>
          </div>

          {/* Backend Status indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', color: serverHealth?.status === 'healthy' ? '#34d399' : '#f87171' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: serverHealth?.status === 'healthy' ? '#10b981' : '#ef4444', display: 'inline-block' }} />
            {serverHealth?.status === 'healthy' ? 'API Online' : 'API Offline'}
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button 
          className={`nav-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          <MessageSquare size={17} />
          AI Copilot
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'claims' ? 'active' : ''}`}
          onClick={() => setActiveTab('claims')}
        >
          <FileText size={17} />
          Claims Registry & SIU
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'preclaim' ? 'active' : ''}`}
          onClick={() => setActiveTab('preclaim')}
        >
          <Activity size={17} />
          Pre-Claim Adjudicator
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'accumulator' ? 'active' : ''}`}
          onClick={() => setActiveTab('accumulator')}
        >
          <Calculator size={17} />
          Benefit Accumulator & OOP Max
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'policy' ? 'active' : ''}`}
          onClick={() => setActiveTab('policy')}
        >
          <Layers size={17} />
          Policy RAG Clauses
        </button>
      </nav>

      {/* Main Tab Content */}
      <main className="main-content">
        {/* TAB 1: CHAT COPILOT */}
        {activeTab === 'chat' && (
          <div className="chat-container">
            <div className="chat-main">
              <div className="chat-messages">
                {messages.map((m, idx) => (
                  <div key={idx} className={`message-bubble ${m.role}`}>
                    <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>

                    {m.role === 'assistant' && (
                      <div className="message-meta">
                        {m.grounding_score !== undefined && (
                          <span className={`badge-grounding ${m.grounding_score >= 0.5 ? 'high' : 'low'}`}>
                            🛡️ Grounding: {Math.round(m.grounding_score * 100)}%
                          </span>
                        )}
                        {m.model_used && (
                          <span style={{ color: 'var(--text-dim)' }}>
                            Model: {m.model_used}
                          </span>
                        )}
                        {m.latency_ms && (
                          <span style={{ color: 'var(--text-dim)' }}>
                            {Math.round(m.latency_ms)}ms
                          </span>
                        )}
                      </div>
                    )}

                    {m.citations && m.citations.length > 0 && (
                      <div className="citations-list">
                        {m.citations.map((c, cIdx) => (
                          <span key={cIdx} className="citation-chip">
                            📌 {c}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                {isLoadingChat && (
                  <div className="message-bubble assistant">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
                      <span className="spinner" /> Analyzing policy clauses & claims registry...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="chat-input-area">
                <div className="quick-prompts">
                  <button 
                    className="quick-prompt-chip"
                    onClick={() => handleSendMessage("What is the status of claim CLM1004?")}
                  >
                    🔍 Status of Claim CLM1004
                  </button>
                  <button 
                    className="quick-prompt-chip"
                    onClick={() => handleSendMessage("Can you generate an appeal letter for it?")}
                  >
                    📝 Appeal letter for active claim
                  </button>
                  <button 
                    className="quick-prompt-chip"
                    onClick={() => handleSendMessage("Is cosmetic rhinoplasty covered under Section 4.1?")}
                  >
                    🏥 Is cosmetic rhinoplasty covered?
                  </button>
                  <button 
                    className="quick-prompt-chip"
                    onClick={() => handleSendMessage("Calculate my estimated reimbursement for a $3,500 emergency room bill")}
                  >
                    💰 Calculate $3,500 reimbursement
                  </button>
                </div>

                <div className="input-box-wrapper">
                  <input
                    type="text"
                    className="chat-input"
                    placeholder="Ask about policy benefits, check a claim ID, calculate coinsurance, or dispute a denial..."
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    disabled={isLoadingChat}
                  />
                  <button 
                    className="send-btn"
                    onClick={() => handleSendMessage()}
                    disabled={isLoadingChat || !inputMessage.trim()}
                  >
                    <Send size={16} /> Send
                  </button>
                </div>
              </div>
            </div>

            {/* Sidebar Inspector */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="glass-card">
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-emerald)', marginBottom: '0.5rem' }}>
                  <Shield size={18} /> Active HIPAA Profile
                </h4>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                  Logged in as: <strong style={{ color: 'var(--text-main)' }}>{currentIdentity.name}</strong><br />
                  Member ID: <code>{currentIdentity.id || 'UNAUTHENTICATED'}</code><br />
                  Permissions: <span style={{ color: currentIdentity.role === 'admin' ? '#34d399' : '#60a5fa', fontWeight: 600 }}>{currentIdentity.role.toUpperCase()}</span>
                </div>
                <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.5rem' }}>
                  Switch identities above to see Protected Health Information (PHI) access control in action.
                </div>
              </div>

              <div className="glass-card">
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>
                  <TrendingUp size={18} /> Plan Parameters
                </h4>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                  <div>Individual Deductible: <strong>$1,000</strong></div>
                  <div>Family Deductible: <strong>$2,500</strong></div>
                  <div>In-Net Coinsurance: <strong>80%</strong></div>
                  <div>Out-of-Net Coinsurance: <strong>70%</strong></div>
                  <div>Individual OOP Max: <strong>$4,500</strong></div>
                  <div>Filing Deadline: <strong>90 Days</strong></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: CLAIMS REGISTRY & SIU */}
        {activeTab === 'claims' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <h2>ApexShield Claims Database</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Persistent SQLite claims registry with HIPAA Protected Health Information (PHI) guard & SIU anomaly detector.
                </p>
              </div>
              <input 
                type="text"
                className="chat-input"
                style={{ maxWidth: 300 }}
                placeholder="Search by ID, Name, Category..."
                value={claimsSearch}
                onChange={(e) => setClaimsSearch(e.target.value)}
              />
            </div>

            <div className="claims-grid">
              {claims
                .filter(c => !claimsSearch || `${c.claim_id} ${c.policyholder_name} ${c.claim_type}`.toLowerCase().includes(claimsSearch.toLowerCase()))
                .map((claim) => {
                  const authorized = isClaimAuthorized(claim);
                  const statusClass = 
                    claim.status === 'Approved' ? 'approved' :
                    claim.status === 'Under Review' ? 'under-review' :
                    claim.status === 'Rejected' ? 'rejected' : 'docs-needed';

                  return (
                    <div key={claim.claim_id} className={`claim-card ${!authorized ? 'restricted' : ''}`}>
                      <div className="claim-header">
                        <div>
                          <div className="claim-id">{claim.claim_id}</div>
                          <div className="claim-holder">
                            {authorized ? claim.policyholder_name : '🔒 Protected Health Information (PHI)'}
                          </div>
                        </div>
                        <span className={`status-pill ${statusClass}`}>
                          {claim.status}
                        </span>
                      </div>

                      {authorized ? (
                        <>
                          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            Policy ID: <code>{claim.policyholder_id}</code> | Category: <strong>{claim.claim_type}</strong>
                          </div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                            Claimed: ${claim.claim_amount.toLocaleString()} &bull; Approved: ${claim.approved_amount.toLocaleString()}
                          </div>
                          <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            <strong>Next Action:</strong> {claim.next_action}<br />
                            <strong>Adjudication Notes:</strong> {claim.notes}
                          </div>

                          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                            <button 
                              className="btn-action"
                              onClick={() => runAnomalyAudit(claim.claim_id)}
                            >
                              <Search size={14} /> Run SIU Audit
                            </button>
                            <button 
                              className="btn-action primary"
                              onClick={() => runGenerateAppeal(claim.claim_id)}
                            >
                              <FileText size={14} /> Appeal Letter
                            </button>
                          </div>
                        </>
                      ) : (
                        <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(239, 68, 68, 0.25)', color: '#f87171', fontSize: '0.85rem' }}>
                          <Lock size={16} style={{ display: 'inline', marginRight: 6 }} />
                          <strong>HIPAA Access Restricted</strong><br />
                          You are currently signed in as <code>{currentIdentity.id || 'Guest'}</code>. This claim belongs to policyholder <code>{claim.policyholder_id}</code>. Protected Health Information is withheld.
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        )}

        {/* TAB 3: PRE-CLAIM ADJUDICATOR */}
        {activeTab === 'preclaim' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
            <div className="glass-card">
              <h3>⚡ Pre-Claim Adjudication Simulator</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                Simulate prospective medical procedures against policy exclusions (Section 4) and waiting periods (Section 3).
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>Treatment / Procedure Name</label>
                  <input 
                    type="text" 
                    className="chat-input"
                    value={treatmentName}
                    onChange={(e) => setTreatmentName(e.target.value)}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>Estimated Cost ($)</label>
                    <input 
                      type="number" 
                      className="chat-input"
                      value={estimatedCost}
                      onChange={(e) => setEstimatedCost(e.target.value)}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>Months Enrolled</label>
                    <input 
                      type="number" 
                      className="chat-input"
                      value={monthsEnrolled}
                      onChange={(e) => setMonthsEnrolled(e.target.value)}
                    />
                  </div>
                </div>

                <div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                    <input 
                      type="checkbox"
                      checked={isPreExisting}
                      onChange={(e) => setIsPreExisting(e.target.checked)}
                    />
                    Is this treatment for a pre-existing medical condition? (Section 3.2: 12-month waiting period)
                  </label>
                </div>

                <button 
                  className="send-btn" 
                  style={{ width: '100%', justifyContent: 'center', padding: '0.85rem' }}
                  onClick={handlePreclaimAdjudicate}
                  disabled={isLoadingPreclaim}
                >
                  {isLoadingPreclaim ? 'Adjudicating against Policy...' : 'Run Pre-Adjudication Check'}
                </button>
              </div>
            </div>

            <div className="glass-card">
              <h3>Adjudication Determination</h3>
              {preclaimResult ? (
                <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ 
                    padding: '1rem', 
                    borderRadius: 'var(--radius-md)', 
                    background: preclaimResult.is_eligible ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                    border: `1px solid ${preclaimResult.is_eligible ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem'
                  }}>
                    {preclaimResult.is_eligible ? (
                      <CheckCircle2 size={24} color="#10b981" />
                    ) : (
                      <AlertTriangle size={24} color="#ef4444" />
                    )}
                    <div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: preclaimResult.is_eligible ? '#34d399' : '#f87171' }}>
                        {preclaimResult.decision}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                        {preclaimResult.notes}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="glass-card" style={{ padding: '1rem' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Estimated Plan Payout</div>
                      <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#34d399' }}>
                        ${preclaimResult.estimated_reimbursement.toLocaleString()}
                      </div>
                    </div>
                    <div className="glass-card" style={{ padding: '1rem' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Patient Responsibility</div>
                      <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f87171' }}>
                        ${preclaimResult.patient_estimated_share.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  <div>
                    <h5 style={{ marginBottom: '0.4rem', color: 'var(--text-muted)' }}>Governing Policy Sections:</h5>
                    <div className="citations-list">
                      {preclaimResult.governing_sections.map((sec, sIdx) => (
                        <span key={sIdx} className="citation-chip">{sec}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '3rem 0' }}>
                  Fill in procedure details on the left and click Run to simulate coverage determination.
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 4: BENEFIT ACCUMULATOR */}
        {activeTab === 'accumulator' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
            <div className="glass-card">
              <h3>💳 Benefit Accumulator State Machine</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                Section 5.1 Deductible & Section 5.2 Catastrophic Out-of-Pocket Maximum state tracking (Persistent in SQLite).
              </p>

              {accumulatorData && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  {/* Deductible Progress */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem' }}>
                      <span>Annual Deductible ($1,000 Ind)</span>
                      <strong>${1000 - accumulatorData.deductible_remaining_after} / $1,000</strong>
                    </div>
                    <div className="progress-container">
                      <div className="progress-bar-bg">
                        <div 
                          className="progress-bar-fill" 
                          style={{ width: `${Math.min(100, ((1000 - accumulatorData.deductible_remaining_after) / 1000) * 100)}%` }}
                        />
                      </div>
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                      Remaining Deductible: <strong>${accumulatorData.deductible_remaining_after.toFixed(2)}</strong>
                    </div>
                  </div>

                  {/* OOP Max Progress */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem' }}>
                      <span>Out-of-Pocket Maximum ($4,500 Ind Cap)</span>
                      <strong>${accumulatorData.oop_accumulated_total.toFixed(2)} / $4,500</strong>
                    </div>
                    <div className="progress-container">
                      <div className="progress-bar-bg">
                        <div 
                          className="progress-bar-fill" 
                          style={{ width: `${Math.min(100, (accumulatorData.oop_accumulated_total / 4500) * 100)}%`, background: 'linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)' }}
                        />
                      </div>
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                      Remaining before 100% Plan Coverage: <strong>${accumulatorData.oop_max_remaining.toFixed(2)}</strong>
                    </div>
                  </div>

                  {accumulatorData.catastrophic_100_percent_active && (
                    <div style={{ padding: '0.85rem', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: 'var(--radius-md)', color: '#34d399', fontSize: '0.85rem' }}>
                      🎉 <strong>Catastrophic Coverage Activated per Section 5.2!</strong> Plan pays 100% of eligible charges with $0 patient responsibility.
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="glass-card">
              <h3>Simulate New Medical Expense</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
                Process a claim through the accumulator state machine for <strong>{currentIdentity.name}</strong>.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>Eligible Medical Expense ($)</label>
                  <input 
                    type="number"
                    className="chat-input"
                    value={simulatedCharge}
                    onChange={(e) => setSimulatedCharge(e.target.value)}
                  />
                </div>

                <button 
                  className="send-btn"
                  style={{ width: '100%', justifyContent: 'center', padding: '0.85rem' }}
                  onClick={handleAccumulateExpense}
                >
                  Adjudicate & Accumulate to SQLite
                </button>

                {accumulatorData && (
                  <div style={{ marginTop: '0.75rem', padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-md)', fontSize: '0.85rem' }}>
                    <div style={{ fontWeight: 600, marginBottom: '0.5rem', color: 'var(--accent-emerald)' }}>Last Adjudication Breakdown:</div>
                    <div>Eligible Charges: <strong>${accumulatorData.eligible_charges.toFixed(2)}</strong></div>
                    <div>Deductible Satisfied: <strong>${accumulatorData.deductible_applied.toFixed(2)}</strong></div>
                    <div>Plan Payout: <strong style={{ color: '#34d399' }}>${accumulatorData.plan_payout.toFixed(2)}</strong></div>
                    <div>Patient Responsibility: <strong style={{ color: '#f87171' }}>${accumulatorData.patient_responsibility.toFixed(2)}</strong></div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: POLICY RAG EXPLORER */}
        {activeTab === 'policy' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <div>
                <h2>ApexShield Care Plus Policy Catalog</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  All 27 granular policy clauses with BM25Okapi lexical matching, synonym expansion, and Reciprocal Rank Fusion (RRF).
                </p>
              </div>

              <form onSubmit={handlePolicySearch} style={{ display: 'flex', gap: '0.5rem' }}>
                <input 
                  type="text" 
                  className="chat-input" 
                  placeholder="Search policy clauses..."
                  value={policyQuery}
                  onChange={(e) => setPolicyQuery(e.target.value)}
                  style={{ width: 280 }}
                />
                <button type="submit" className="send-btn">
                  <Search size={15} /> Search
                </button>
              </form>
            </div>

            {policyResults.length > 0 && (
              <div style={{ marginBottom: '2rem' }}>
                <h4 style={{ color: 'var(--accent-emerald)', marginBottom: '0.75rem' }}>
                  🎯 BM25 + RRF Search Results ({policyResults.length} clauses matched)
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {policyResults.map((res, rIdx) => (
                    <div key={rIdx} className="glass-card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                        <span className="citation-chip" style={{ fontSize: '0.85rem' }}>{res.citation}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>RRF Score: {res.score?.toFixed(4)}</span>
                      </div>
                      <p style={{ fontSize: '0.92rem', color: 'var(--text-main)', lineHeight: '1.6' }}>{res.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <h4 style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Full Catalog ({allSections.length} Indexed Sections)
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: '1rem' }}>
              {allSections.map((sec, sIdx) => (
                <div key={sIdx} className="glass-card" style={{ padding: '1.1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                    <span className="citation-chip">{sec.citation}</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>{sec.category}</span>
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.4rem' }}>
                    {sec.section_title}
                  </div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                    {sec.content.length > 180 ? sec.content.slice(0, 180) + '...' : sec.content}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* SIU Anomaly Modal */}
      {anomalyModal && (
        <div className="modal-overlay" onClick={() => setAnomalyModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <h3>🔍 Special Investigations Unit (SIU) Anomaly Audit</h3>
              <button className="btn-action" onClick={() => setAnomalyModal(null)}>
                <X size={16} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <div className="glass-card" style={{ flex: 1, padding: '1rem' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Risk Level</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700, color: anomalyModal.risk_level === 'HIGH' ? '#f87171' : anomalyModal.risk_level === 'MEDIUM' ? '#fbbf24' : '#34d399' }}>
                    {anomalyModal.risk_level} (Score: {anomalyModal.risk_score.toFixed(2)})
                  </div>
                </div>
                <div className="glass-card" style={{ flex: 1, padding: '1rem' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Statistical Z-Score</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>
                    {anomalyModal.z_score > 0 ? `+${anomalyModal.z_score}` : anomalyModal.z_score}&sigma;
                  </div>
                </div>
                <div className="glass-card" style={{ flex: 1, padding: '1rem' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Timely Filing</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700, color: anomalyModal.timely_filing_ok ? '#34d399' : '#f87171' }}>
                    {anomalyModal.timely_filing_ok ? 'PASS' : 'FAIL'}
                  </div>
                </div>
              </div>

              {anomalyModal.is_potential_duplicate && (
                <div style={{ padding: '0.85rem', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: 'var(--radius-md)', color: '#f87171' }}>
                  🚨 <strong>Potential Duplicate Billing Flagged!</strong> This claim matches prior submissions ({anomalyModal.duplicate_claim_ids.join(', ')}) filed for this member.
                </div>
              )}

              <div>
                <h5 style={{ marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Audited Flags:</h5>
                {anomalyModal.flags.length > 0 ? (
                  <ul style={{ paddingLeft: '1.25rem', fontSize: '0.88rem', color: '#fca5a5', lineHeight: '1.6' }}>
                    {anomalyModal.flags.map((f, fIdx) => (
                      <li key={fIdx}>{f}</li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ color: '#34d399', fontSize: '0.88rem' }}>Clean claim record. Zero statistical or policy anomalies found.</div>
                )}
              </div>

              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: 'var(--radius-md)', fontSize: '0.85rem' }}>
                <strong>SIU Recommendation:</strong> {anomalyModal.recommendation}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Appeal Letter Modal */}
      {appealModal && (
        <div className="modal-overlay" onClick={() => setAppealModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <h3>Form CL-01 Notice of Claim Dispute & Appeal</h3>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  className="btn-action"
                  onClick={() => navigator.clipboard.writeText(appealModal.letter_content)}
                >
                  <Copy size={14} /> Copy
                </button>
                <button className="btn-action" onClick={() => setAppealModal(null)}>
                  <X size={16} />
                </button>
              </div>
            </div>

            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Statutory Deadline: <strong>{appealModal.statutory_deadline}</strong> | Submission Email: <code>{appealModal.appeal_channel}</code>
            </div>

            <pre>{appealModal.letter_content}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import load_vector_store, retrieve_context
from src.llm_handler import get_claude_response
from src.visualizer import get_chart_for_query

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFlow AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  font-family: 'Inter', sans-serif;
  background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f1a 100%) !important;
  background-attachment: fixed !important;
  color: #e2e8f0;
}

/* Animated subtle noise overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    radial-gradient(ellipse at 20% 20%, rgba(124,58,237,0.07) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(6,182,212,0.07) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(37,99,235,0.04) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: rgba(10,10,20,0.85) !important;
  backdrop-filter: blur(24px) !important;
  border-right: 1px solid rgba(124,58,237,0.25) !important;
  box-shadow: 4px 0 40px rgba(124,58,237,0.08) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* Sidebar title */
.sb-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 4px 12px 4px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  margin-bottom: 20px;
}
.sb-brand-icon {
  font-size: 1.8rem;
  filter: drop-shadow(0 0 8px rgba(124,58,237,0.8));
}
.sb-brand-text { line-height: 1.2; }
.sb-brand-title {
  font-size: 1.1rem;
  font-weight: 700;
  background: linear-gradient(90deg, #a78bfa, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sb-brand-sub {
  font-size: 0.7rem;
  color: #64748b;
  letter-spacing: 0.05em;
}

/* Section headings inside sidebar */
.sb-section-title {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #475569;
  margin: 20px 0 10px 0;
}

/* Pipeline status rows */
.status-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 6px;
  font-size: 0.82rem;
  color: #94a3b8;
}
.pulse-dot {
  width: 8px;
  height: 8px;
  min-width: 8px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse-green 2s infinite;
}
.pulse-dot.err {
  background: #ef4444;
  animation: pulse-red 2s infinite;
}
@keyframes pulse-green {
  0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
  70%  { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
  100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}
@keyframes pulse-red {
  0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
  70%  { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}

/* Dataset stat rows */
.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  margin-bottom: 5px;
  font-size: 0.82rem;
}
.stat-label { color: #64748b; }
.stat-value {
  font-weight: 600;
  background: linear-gradient(90deg, #a78bfa, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Sample question buttons */
.stButton > button {
  background: rgba(124,58,237,0.12) !important;
  border: 1px solid rgba(124,58,237,0.35) !important;
  border-radius: 10px !important;
  color: #c4b5fd !important;
  width: 100% !important;
  text-align: left !important;
  font-size: 0.8rem !important;
  padding: 9px 13px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 400 !important;
  transition: all 0.2s ease !important;
  margin-bottom: 2px !important;
  line-height: 1.4 !important;
  white-space: normal !important;
  height: auto !important;
}
.stButton > button:hover {
  background: rgba(124,58,237,0.28) !important;
  border-color: rgba(6,182,212,0.55) !important;
  color: #67e8f9 !important;
  box-shadow: 0 0 18px rgba(124,58,237,0.25) !important;
  transform: translateX(2px) !important;
}
.clear-btn > button {
  background: rgba(239,68,68,0.1) !important;
  border-color: rgba(239,68,68,0.3) !important;
  color: #fca5a5 !important;
  border-radius: 10px !important;
}
.clear-btn > button:hover {
  background: rgba(239,68,68,0.22) !important;
  border-color: rgba(239,68,68,0.6) !important;
  color: #fca5a5 !important;
  box-shadow: 0 0 18px rgba(239,68,68,0.2) !important;
}

/* ── Main header ── */
.hero-banner {
  position: relative;
  padding: 36px 40px;
  border-radius: 20px;
  background: linear-gradient(135deg,
    rgba(124,58,237,0.18) 0%,
    rgba(37,99,235,0.14) 50%,
    rgba(6,182,212,0.12) 100%);
  border: 1px solid rgba(124,58,237,0.3);
  box-shadow:
    0 0 60px rgba(124,58,237,0.1),
    inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(16px);
  margin-bottom: 28px;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(124,58,237,0.2) 0%, transparent 70%);
  pointer-events: none;
}
.hero-banner::after {
  content: '';
  position: absolute;
  bottom: -40px; left: -40px;
  width: 160px; height: 160px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.hero-title {
  font-size: 3.5rem;
  font-weight: 900;
  letter-spacing: -1px;
  line-height: 1.1;
  background: linear-gradient(
    90deg,
    #7c3aed 0%,
    #2563eb 40%,
    #06b6d4 70%,
    #10b981 100%
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: none;
  filter: drop-shadow(0 0 30px rgba(124,58,237,0.5));
  margin: 0 0 10px 0;
}
.subtitle {
  font-size: 1.1rem;
  color: rgba(148,163,184,0.8);
  font-weight: 300;
  letter-spacing: 0.5px;
  margin: 0;
}
.hero-pills {
  display: flex;
  gap: 8px;
  margin-top: 16px;
  flex-wrap: wrap;
}
.hero-pill {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid;
}
.pill-purple {
  color: #a78bfa;
  border-color: rgba(167,139,250,0.35);
  background: rgba(124,58,237,0.12);
}
.pill-blue {
  color: #60a5fa;
  border-color: rgba(96,165,250,0.35);
  background: rgba(37,99,235,0.12);
}
.pill-cyan {
  color: #22d3ee;
  border-color: rgba(34,211,238,0.35);
  background: rgba(6,182,212,0.12);
}

/* Gradient divider */
.grad-divider {
  height: 1px;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(124,58,237,0.5) 30%,
    rgba(6,182,212,0.5) 70%,
    transparent 100%);
  margin: 4px 0 28px 0;
  border: none;
}

/* ── Metric cards ── */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}
.metric-card {
  background: rgba(124,58,237,0.08);
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: 16px;
  padding: 22px 20px;
  text-align: center;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, #7c3aed, #2563eb, #06b6d4);
  opacity: 0.6;
}
.metric-card:hover {
  border-color: rgba(6,182,212,0.5);
  box-shadow: 0 0 32px rgba(6,182,212,0.15);
  transform: translateY(-2px);
}
.metric-icon {
  font-size: 2rem;
  margin-bottom: 8px;
  filter: drop-shadow(0 0 10px rgba(124,58,237,0.6));
}
.metric-value {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -1px;
  line-height: 1;
  margin-bottom: 6px;
}
.metric-label {
  font-size: 0.75rem;
  color: rgba(148,163,184,0.6);
  font-weight: 500;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 4px;
}

/* ── Chat container ── */
.chat-wrapper {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 24px;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 40px rgba(0,0,0,0.3);
  margin-bottom: 16px;
  min-height: 200px;
}
.chat-empty {
  text-align: center;
  padding: 48px 24px;
  color: #334155;
}
.chat-empty-icon {
  font-size: 3rem;
  margin-bottom: 12px;
  filter: drop-shadow(0 0 12px rgba(124,58,237,0.3));
}
.chat-empty-text {
  font-size: 1rem;
  color: #475569;
  font-weight: 500;
}
.chat-empty-hint {
  font-size: 0.82rem;
  color: #334155;
  margin-top: 6px;
}

/* Message rows */
.msg-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 20px;
}
.msg-row.user-row {
  flex-direction: row-reverse;
}

/* Avatars */
.avatar {
  width: 36px;
  height: 36px;
  min-width: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: 700;
  flex-shrink: 0;
}
.avatar-user {
  background: linear-gradient(135deg, #7c3aed, #2563eb);
  box-shadow: 0 0 12px rgba(124,58,237,0.45);
  color: white;
}
.avatar-ai {
  background: rgba(6,182,212,0.15);
  border: 1px solid rgba(6,182,212,0.4);
  color: #22d3ee;
  box-shadow: 0 0 12px rgba(6,182,212,0.2);
}

/* Bubble wrappers */
.bubble-wrap { max-width: 78%; }
.bubble-wrap.user-wrap { margin-left: auto; }

/* Sender labels */
.sender-label {
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  margin-bottom: 5px;
  opacity: 0.55;
}
.sender-label.user-lbl { text-align: right; }

/* Bubbles */
.user-bubble {
  background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
  border-radius: 18px 18px 4px 18px;
  padding: 13px 18px;
  color: #ede9fe;
  font-size: 0.92rem;
  line-height: 1.6;
  box-shadow: 0 4px 20px rgba(124,58,237,0.35);
  word-wrap: break-word;
}
.ai-bubble {
  background: rgba(6,182,212,0.07);
  border: 1px solid rgba(6,182,212,0.28);
  border-radius: 18px 18px 18px 4px;
  padding: 13px 18px;
  color: #e2e8f0;
  font-size: 0.92rem;
  line-height: 1.7;
  box-shadow: 0 4px 20px rgba(6,182,212,0.08);
  word-wrap: break-word;
}

/* ── Chat input ── */
.stChatInput {
  border-radius: 14px !important;
}
.stChatInput > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(124,58,237,0.35) !important;
  border-radius: 14px !important;
  box-shadow: 0 0 0 0 rgba(124,58,237,0);
  transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}
.stChatInput > div:focus-within {
  border-color: rgba(6,182,212,0.6) !important;
  box-shadow: 0 0 24px rgba(124,58,237,0.18) !important;
}
.stChatInput textarea {
  color: #e2e8f0 !important;
  font-family: 'Inter', sans-serif !important;
}
.stChatInput button {
  background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
  border-radius: 10px !important;
  border: none !important;
}
.stChatInput button:hover {
  box-shadow: 0 0 16px rgba(124,58,237,0.5) !important;
}

/* ── Spinner override ── */
.stSpinner > div {
  border-top-color: #7c3aed !important;
}

/* ── Error / info boxes ── */
.stAlert {
  border-radius: 12px !important;
  backdrop-filter: blur(10px) !important;
}

/* ── Plotly chart container ── */
.stPlotlyChart {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.07);
  background: rgba(255,255,255,0.02);
  margin-top: 12px;
}

/* Hide default streamlit top bar decorations */
#MainMenu, footer, header { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(124,58,237,0.35);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(6,182,212,0.5); }
</style>
""", unsafe_allow_html=True)


# ─── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        index, documents = load_vector_store("data/processed/")
        return index, documents, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def load_dataframe():
    try:
        df = pd.read_csv("data/processed/bookings_clean.csv")
        return df, None
    except Exception as e:
        return None, str(e)


# ─── Load resources ───────────────────────────────────────────────────────────
index, documents, pipeline_error = load_pipeline()
df, df_error = load_dataframe()

# ─── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-icon">✈️</div>
      <div class="sb-brand-text">
        <div class="sb-brand-title">SmartFlow AI</div>
        <div class="sb-brand-sub">Airline Intelligence Platform</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline status
    st.markdown('<div class="sb-section-title">Pipeline Status</div>', unsafe_allow_html=True)

    def _status_row(label, ok):
        dot_cls = "pulse-dot" if ok else "pulse-dot err"
        return (
            f'<div class="status-row">'
            f'<div class="{dot_cls}"></div>'
            f'<span>{label}</span>'
            f'</div>'
        )

    st.markdown(
        _status_row("Vector Store", index is not None) +
        _status_row("Dataset", df is not None) +
        _status_row("Claude API", bool(os.getenv("ANTHROPIC_API_KEY"))),
        unsafe_allow_html=True,
    )

    # Dataset stats
    if df is not None:
        st.markdown('<div class="sb-section-title">Dataset Stats</div>', unsafe_allow_html=True)
        sat_rate = (df["satisfaction"] == "satisfied").mean() * 100

        def _stat(label, value):
            return (
                f'<div class="stat-row">'
                f'<span class="stat-label">{label}</span>'
                f'<span class="stat-value">{value}</span>'
                f'</div>'
            )

        st.markdown(
            _stat("Records", f"{len(df):,}") +
            _stat("Satisfaction", f"{sat_rate:.1f}%") +
            _stat("Classes", str(df["Class"].nunique())) +
            _stat("Avg Age", f"{df['Age'].mean():.1f} yrs") +
            _stat("Avg Distance", f"{df['Flight Distance'].mean():,.0f} mi"),
            unsafe_allow_html=True,
        )

    # Sample questions
    st.markdown('<div class="sb-section-title">Try a Question</div>', unsafe_allow_html=True)
    sample_questions = [
        "What factors most influence passenger satisfaction?",
        "Show me a satisfaction breakdown by travel class",
        "How do delays affect passenger satisfaction?",
        "What is the age distribution of satisfied passengers?",
        "Compare business vs personal travel satisfaction rates",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sq_{q[:20]}"):
            st.session_state.pending_question = q

    st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.pending_question = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">✈️ SmartFlow AI</p>
  <p class="subtitle">
    Ask anything about airline passenger satisfaction —
    powered by RAG + Claude Sonnet
  </p>
  <div class="hero-pills">
    <span class="hero-pill pill-purple">RAG Pipeline</span>
    <span class="hero-pill pill-blue">Claude Sonnet</span>
    <span class="hero-pill pill-cyan">103K Records</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Startup errors ───────────────────────────────────────────────────────────
if pipeline_error:
    st.error(f"Vector store failed to load: {pipeline_error}")
if df_error:
    st.error(f"Dataset failed to load: {df_error}")

# ─── Metrics row ──────────────────────────────────────────────────────────────
if df is not None:
    sat_rate = (df["satisfaction"] == "satisfied").mean() * 100
    avg_dist = df["Flight Distance"].mean()

    st.markdown(f"""
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-value" style="color:#7c3aed;">{len(df):,}</div>
        <div class="metric-label">Total Records</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">⭐</div>
        <div class="metric-value" style="color:#06b6d4;">{sat_rate:.1f}%</div>
        <div class="metric-label">Satisfaction Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">🛫</div>
        <div class="metric-value" style="color:#2563eb;">{avg_dist:,.0f}</div>
        <div class="metric-label">Avg Flight Distance (mi)</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">🎫</div>
        <div class="metric-value" style="color:#10b981;">{df["Class"].nunique()}</div>
        <div class="metric-label">Travel Classes</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Chat history display ─────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center; padding: 44px 24px 36px;">
      <div style="font-size:3rem; margin-bottom:16px;
                  filter: drop-shadow(0 0 16px rgba(124,58,237,0.5));">✈️</div>
      <div style="font-size:1.45rem; font-weight:700; margin-bottom:10px;
                  background: linear-gradient(90deg,#7c3aed,#2563eb,#06b6d4);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text;">
        Welcome to SmartFlow AI
      </div>
      <div style="font-size:0.88rem; color:#64748b; margin-bottom:18px;">
        Powered by Claude Sonnet + Voyage AI + RAG Architecture
      </div>
      <div style="height:1px;
                  background:linear-gradient(90deg,transparent,rgba(124,58,237,0.4),rgba(6,182,212,0.4),transparent);
                  margin: 0 auto 18px; max-width:320px;"></div>
      <div style="font-size:0.9rem; color:#475569; margin-bottom:22px;">
        Ask me anything about <strong style="color:#94a3b8;">103,594</strong> airline passenger records
      </div>
      <div style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
        <span style="background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.4);
                     border-radius:20px; padding:8px 16px; color:#c4b5fd;
                     font-size:0.82rem; cursor:pointer;">✈️ Satisfaction by class</span>
        <span style="background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.4);
                     border-radius:20px; padding:8px 16px; color:#c4b5fd;
                     font-size:0.82rem; cursor:pointer;">⏱️ Delay analysis</span>
        <span style="background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.4);
                     border-radius:20px; padding:8px 16px; color:#c4b5fd;
                     font-size:0.82rem; cursor:pointer;">👥 Passenger demographics</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user-row">
              <div class="avatar avatar-user">U</div>
              <div class="bubble-wrap user-wrap">
                <div class="sender-label user-lbl">You</div>
                <div class="user-bubble">{turn["content"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            display_text = (
                turn["content"]
                .replace("SHOW_CHART\n", "")
                .replace("SHOW_CHART", "")
                .strip()
            )
            # Convert newlines to <br> for HTML rendering
            display_text_html = display_text.replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-row">
              <div class="avatar avatar-ai">AI</div>
              <div class="bubble-wrap">
                <div class="sender-label">SmartFlow AI</div>
                <div class="ai-bubble">{display_text_html}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            # Re-render chart for this historical turn
            if turn.get("has_chart") and df is not None:
                fig = get_chart_for_query(turn.get("query", ""), df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

# ─── Chat input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("pending_question", "") or ""

user_input = st.chat_input(
    placeholder="Ask about passenger satisfaction, delays, class breakdowns, service ratings…",
)

if prefill and not user_input:
    user_input = prefill

# ─── Response generation ──────────────────────────────────────────────────────
if user_input and user_input.strip():
    query = user_input.strip()

    if not index or not documents:
        st.error("Vector store is not loaded. Cannot process questions.")
    else:
        # Immediately show the user bubble
        st.markdown(f"""
        <div class="msg-row user-row">
          <div class="avatar avatar-user">U</div>
          <div class="bubble-wrap user-wrap">
            <div class="sender-label user-lbl">You</div>
            <div class="user-bubble">{query}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("SmartFlow AI is thinking…"):
            try:
                context = retrieve_context(query, index, documents, k=5)
                clean_history = [
                    {"role": t["role"], "content": t["content"]}
                    for t in st.session_state.chat_history
                ]
                raw_response = get_claude_response(query, context, clean_history)
            except Exception as e:
                raw_response = f"Error generating response: {str(e)}"

        wants_chart = raw_response.startswith("SHOW_CHART")
        display_text = (
            raw_response
            .replace("SHOW_CHART\n", "")
            .replace("SHOW_CHART", "")
            .strip()
        )
        display_text_html = display_text.replace("\n", "<br>")

        st.markdown(f"""
        <div class="msg-row">
          <div class="avatar avatar-ai">AI</div>
          <div class="bubble-wrap">
            <div class="sender-label">SmartFlow AI</div>
            <div class="ai-bubble">{display_text_html}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if wants_chart and df is not None:
            fig = get_chart_for_query(query, df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Persist to session history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": raw_response,
            "has_chart": wants_chart,
            "query": query,
        })

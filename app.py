import os
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="BITS Pilani | Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLLEGE_NAME = "BITS Pilani"
TAGLINE = "Student Knowledge Assistant"
CAMPUS_IMAGE = (
    "https://images.unsplash.com/photo-1562774053-701939374585"
    "?auto=format&fit=crop&w=2000&q=85"
)
PROGRAMMES_BY_LEVEL = {
    "UG": [
        "B.E. Chemical Engineering", "B.E. Civil Engineering", "B.E. Computer Science",
        "B.E. Electrical and Electronics Engineering", "B.E. Electronics and Communication Engineering",
        "B.E. Electronics and Instrumentation Engineering", "B.E. Mechanical Engineering",
        "B.E. Manufacturing Engineering", "B.Pharm.", "B.Sc. Biological Sciences",
        "B.Sc. Chemistry", "B.Sc. Economics", "B.Sc. Mathematics", "B.Sc. Physics",
    ],
    "PG": [
        "M.E. Biotechnology", "M.E. Chemical Engineering", "M.E. Civil Engineering",
        "M.E. Communication Engineering", "M.E. Computer Science", "M.E. Embedded Systems",
        "M.E. Manufacturing Systems Engineering", "M.E. Mechanical Engineering",
        "M.E. Microelectronics", "M.E. Software Systems", "M.Pharm.", "MBA",
    ],
    "PhD": [
        "PhD in Biological Sciences", "PhD in Chemical Engineering", "PhD in Chemistry",
        "PhD in Civil Engineering", "PhD in Computer Science", "PhD in Economics and Finance",
        "PhD in Electrical and Electronics Engineering", "PhD in Humanities and Social Sciences",
        "PhD in Management", "PhD in Mathematics", "PhD in Mechanical Engineering",
        "PhD in Pharmacy", "PhD in Physics",
    ],
}

# ----------------------------- Styling -----------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,500&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --cream: #faf6ee;
  --paper: #fffdf8;
  --maroon: #6d1a2b;
  --maroon-deep: #4a1120;
  --rose: #f4e6e6;
  --gold: #b8923f;
  --ink: #2b1d20;
  --muted: #7b6b68;
  --line: #e7dcc9;
  color-scheme: light;
}

/* Force a light, readable theme even if Streamlit/browser is in dark mode */
.stApp { background: var(--cream); color: var(--ink); font-family: 'Source Sans 3', sans-serif; }
.stApp p, .stApp label, .stApp li, .stApp span,
.stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1360px; }

/* ---------- Top bar ---------- */
.topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 10px 4px 16px; border-bottom: 1px solid var(--line); margin-bottom: 20px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 42px; height: 42px; border-radius: 50%; background: var(--maroon);
  display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.brand-name { font-family: 'Newsreader', serif; font-size: 22px; font-weight: 700; color: var(--maroon-deep); line-height: 1.1; }
.brand-tag { font-size: 12px; color: var(--muted); margin-top: 2px; }
.nav-links { display: flex; gap: 26px; font-weight: 600; font-size: 14px; color: var(--ink); }
.nav-links span:first-child { color: var(--maroon); border-bottom: 2px solid var(--gold); padding-bottom: 2px; }
.portal {
  background: var(--maroon); color: #fff !important; border-radius: 999px;
  padding: 9px 18px; font-weight: 600; font-size: 14px; white-space: nowrap;
}

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 18px; min-height: 430px;
  background-image: linear-gradient(100deg, rgba(58,12,26,.92) 0%, rgba(74,17,32,.78) 45%, rgba(74,17,32,.30) 100%), url('__CAMPUS_IMAGE__');
  background-size: cover; background-position: center;
  padding: 56px 48px; display: flex; align-items: flex-end;
}
.hero-copy { max-width: 560px; }
.hero h1 {
  font-family: 'Newsreader', serif; font-weight: 600; color: #fff !important;
  font-size: clamp(38px, 4.6vw, 62px); line-height: 1.04; margin: 0 0 14px; letter-spacing: -0.01em;
}
.hero p { color: #f3e6e2 !important; font-size: 18px; line-height: 1.55; margin: 0; max-width: 470px; }

/* ---------- Help topics ---------- */
.topics-title { font-family: 'Newsreader', serif; font-size: 28px; font-weight: 600; color: var(--maroon-deep); margin: 28px 0 4px; }
.topics-sub { color: var(--muted); font-size: 15px; margin-bottom: 14px; }
.topic-list { border-top: 1px solid var(--line); }
.topic { display: flex; gap: 16px; align-items: baseline; padding: 14px 2px; border-bottom: 1px solid var(--line); }
.topic b { font-family: 'Newsreader', serif; font-size: 19px; font-weight: 600; color: var(--maroon); min-width: 190px; }
.topic span { color: #5a4a47; font-size: 15px; line-height: 1.5; }
.fineprint { color: var(--muted); font-size: 12px; margin-top: 14px; }

/* ---------- Chat card (right column) ---------- */
[data-testid="stColumn"]:has(.st-key-chat_card) { position: sticky; top: 1rem; align-self: flex-start; }
.st-key-chat_card {
  background: var(--paper); border: 1px solid var(--line) !important; border-radius: 18px;
  box-shadow: 0 10px 34px rgba(74,17,32,.10); padding: 0 !important; overflow: hidden;
}
.chat-head { background: var(--maroon); padding: 16px 18px; display: flex; align-items: center; gap: 12px; }
.chat-avatar {
  width: 40px; height: 40px; border-radius: 50%; background: var(--cream);
  display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.chat-title { font-family: 'Newsreader', serif; color: #fff !important; font-size: 19px; font-weight: 600; line-height: 1.2; }
.chat-status { color: #ecd9d6 !important; font-size: 12px; margin-top: 2px; }
.chat-status i { color: #7be0a4; font-style: normal; }

.st-key-chat_controls { padding: 10px 14px 0; }
.st-key-chat_controls [data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 12px; font-weight: 600; }

.st-key-chat_history { padding: 4px 8px; }
.empty-state { padding: 10px 6px 4px; }
.empty-state b { font-family: 'Newsreader', serif; font-size: 20px; color: var(--maroon-deep); font-weight: 600; }
.empty-state p { color: var(--muted) !important; font-size: 14px; margin: 4px 0 8px; }

/* Messages */
div[data-testid="stChatMessage"] {
  background: #fff !important; border: 1px solid var(--line); border-radius: 14px;
  padding: 10px 12px; margin-bottom: 8px; color: var(--ink) !important;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--rose) !important; border-color: #ead2d3;
}
div[data-testid="stChatMessage"] * { color: var(--ink) !important; font-size: 15px; line-height: 1.55; }

/* Buttons (quick questions, clear) */
div.stButton > button {
  background: #fff !important; color: var(--maroon) !important; border: 1px solid var(--line) !important;
  border-radius: 10px; font-weight: 600; min-height: 42px; white-space: normal; font-size: 14px;
}
div.stButton > button:hover { background: var(--rose) !important; border-color: var(--maroon) !important; }
div.stButton > button:focus-visible { box-shadow: 0 0 0 3px rgba(109,26,43,.22) !important; }

/* Programme select */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #fff !important; border-color: var(--line) !important; }
div[data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--ink) !important; }
div[data-baseweb="popover"] ul { background: #fff !important; }
div[data-baseweb="popover"] li { color: var(--ink) !important; }

/* Composer */
.st-key-chat_composer { padding: 6px 14px 14px; border-top: 1px solid var(--line); background: var(--paper); }
.st-key-chat_composer [data-testid="stForm"] { border: 0; padding: 0; background: transparent; }
.st-key-chat_composer input {
  color: var(--ink) !important; background: #fff !important; border: 1px solid var(--line) !important;
  border-radius: 10px !important;
}
.st-key-chat_composer input:focus { border-color: var(--maroon) !important; box-shadow: 0 0 0 3px rgba(109,26,43,.15) !important; }
.st-key-chat_composer input::placeholder { color: #9a8a86 !important; opacity: 1; }
.st-key-chat_composer button {
  background: var(--maroon) !important; color: #fff !important; border: 0 !important;
  border-radius: 10px !important; min-height: 42px;
}
.st-key-chat_composer button:hover { background: var(--maroon-deep) !important; }
.st-key-chat_composer button p { color: #fff !important; }

/* Fix wide tables inside chat messages */
div[data-testid="stChatMessage"] {
    min-width: 0;
    max-width: 100%;
    overflow: hidden;
}

div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    min-width: 0;
    max-width: 100%;
    overflow-x: auto;
}

/* Make Markdown tables fit the chat width */
div[data-testid="stChatMessage"] table {
    width: 100% !important;
    max-width: 100% !important;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 13px;
}

div[data-testid="stChatMessage"] th,
div[data-testid="stChatMessage"] td {
    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: normal;
    padding: 8px 6px !important;
    vertical-align: top;
}

/* Keep the first column compact and readable */
div[data-testid="stChatMessage"] th:first-child,
div[data-testid="stChatMessage"] td:first-child {
    width: 30%;
}

/* Prevent wide content from expanding the chat card */
.st-key-chat_card {
    min-width: 0;
    max-width: 100%;
}

/* ---------- Mobile ---------- */
@media (max-width: 900px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  [data-testid="stColumn"]:has(.st-key-chat_card) { position: static; }
  .nav-links { display: none; }
  .hero { min-height: 320px; padding: 30px 22px; }
  .hero p { font-size: 16px; }
  .topic { flex-direction: column; gap: 2px; }
  .topic b { min-width: 0; }
  .brand-name { font-size: 18px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""".replace("__CAMPUS_IMAGE__", CAMPUS_IMAGE)

st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------- Backend import -----------------------------
from project import create_assistant

# ----------------------------- Header -----------------------------
st.markdown(
    f"""
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">🏛️</div>
        <div>
          <div class="brand-name">{COLLEGE_NAME}</div>
          <div class="brand-tag">{TAGLINE}</div>
        </div>
      </div>
      <div class="nav-links">
        <span>Home</span><span>Academics</span><span>Admissions</span>
        <span>Student life</span><span>Contact</span>
      </div>
      <div class="portal">Student portal</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Session state -----------------------------
if "programme_level" not in st.session_state:
    st.session_state.programme_level = "UG"
if "programme" not in st.session_state:
    st.session_state.programme = PROGRAMMES_BY_LEVEL["UG"][0]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ----------------------------- Layout: hero left, chat right -----------------------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-copy">
            <h1>Your BITS Pilani journey, made simpler.</h1>
            <p>Ask about programmes, eligibility, academic regulations, admissions and fees.</p>
          </div>
        </div>
        <div class="topics-title">What the assistant can help with</div>
        <div class="topics-sub">Answers should be grounded in available official BITS Pilani academic and fee documents.</div>
        <div class="topic-list">
          <div class="topic"><b>Attendance and exams</b><span>Minimum attendance, exam eligibility, grading and promotion rules.</span></div>
          <div class="topic"><b>Fees and payments</b><span>Tuition, instalments, late charges, refunds and scholarships.</span></div>
          <div class="topic"><b>Course structure</b><span>Credits, summer training and degree requirements for your programme.</span></div>
        </div>
        <div class="fineprint">Programme availability and eligibility can vary by campus and admission year. Verify current details against official BITS Pilani admissions information.</div>
        """,
        unsafe_allow_html=True,
    )

with right:
    with st.container(border=True, key="chat_card"):
        st.markdown(
            """
            <div class="chat-head">
              <div class="chat-avatar">🎓</div>
              <div>
                <div class="chat-title">BITS Pilani assistant</div>
                <div class="chat-status"><i>●</i> Online</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Programme + clear
        with st.container(key="chat_controls"):
            level_col, programme_col = st.columns([1, 2])
            with level_col:
                selected_level = st.selectbox(
                    "Study level", list(PROGRAMMES_BY_LEVEL.keys()),
                    index=list(PROGRAMMES_BY_LEVEL.keys()).index(st.session_state.programme_level),
                    help="UG = undergraduate, PG = postgraduate, PhD = doctoral studies.",
                    key="programme_level_select",
                )
            if selected_level != st.session_state.programme_level:
                st.session_state.programme_level = selected_level
                st.session_state.programme = PROGRAMMES_BY_LEVEL[selected_level][0]
            with programme_col:
                options = PROGRAMMES_BY_LEVEL[st.session_state.programme_level]
                if st.session_state.programme not in options:
                    st.session_state.programme = options[0]
                st.session_state.programme = st.selectbox(
                    "Specific programme", options, index=options.index(st.session_state.programme),
                    help="Select the programme you want answers tailored to.",
                    key="specific_programme_select",
                )
            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.pending_question = None
                st.rerun()

        # Scrollable history (fixed height keeps the composer in place)
        with st.container(height=400, border=False, key="chat_history"):
            if not st.session_state.chat_history:
                st.markdown(
                    '<div class="empty-state"><b>How can I help?</b>'
                    "<p>Choose your level and programme, then ask a question.</p></div>",
                    unsafe_allow_html=True,
                )
                quick_questions = [
                    ("Programmes", "What programmes are available at BITS Pilani?"),
                    ("Eligibility", "What are the eligibility requirements for my programme?"),
                    ("Academic rules", "Explain the academic regulations relevant to my programme."),
                    ("Fees", "What fees apply to my programme?"),
                ]
                qa, qb = st.columns(2)
                for i, (label, question) in enumerate(quick_questions):
                    with (qa if i % 2 == 0 else qb):
                        if st.button(label, key=f"quick_{i}", use_container_width=True):
                            st.session_state.pending_question = question
            for message in st.session_state.chat_history:
                with st.chat_message(
                    message["role"],
                    avatar="🎓" if message["role"] == "assistant" else "🧑",
                ):
                    st.markdown(message["content"])

        # Composer (form, so it stays inside the card instead of page-bottom)
        with st.container(key="chat_composer"):
            with st.form("college_chat_form", clear_on_submit=True, border=False):
                input_col, send_col = st.columns([4, 1.2], vertical_alignment="center")
                with input_col:
                    typed_query = st.text_input(
                        "Message",
                        placeholder="Ask about programmes, eligibility, fees…",
                        label_visibility="collapsed",
                        key="college_message_input",
                    )
                with send_col:
                    submitted = st.form_submit_button("Send", use_container_width=True)

    # Keep newest message in view
    if st.session_state.chat_history:
        components.html(
            """<script>
            const root = window.parent.document.querySelector('.st-key-chat_history');
            if (root) {
              root.querySelectorAll('*').forEach(el => {
                if (el.scrollHeight > el.clientHeight + 4 &&
                    getComputedStyle(el).overflowY !== 'visible') {
                  el.scrollTop = el.scrollHeight;
                }
              });
            }
            </script>""",
            height=0,
        )

    # ----------------------------- Handle input -----------------------------
    user_query = None
    if st.session_state.pending_question:
        user_query = st.session_state.pending_question
        st.session_state.pending_question = None
    elif submitted and typed_query.strip():
        user_query = typed_query.strip()

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        try:
            assistant_app = create_assistant()
            with st.spinner("Checking BITS Pilani documents…"):
                result = assistant_app.invoke({
                    "programme_level": st.session_state.programme_level,
                    "programme": st.session_state.programme,
                    "messages": [("human", user_query)],
                    "query_type": "",
                    "retrieved_context": "",
                })
                answer = result["messages"][-1].content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    "I couldn't process that question. Ensure project.py exposes create_assistant() "
                    "and accepts the programme_level state field, and check your API keys and PDF paths. "
                    f"Details: {exc}"
                ),
            })
        st.rerun()

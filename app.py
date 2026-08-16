import streamlit as st
import pandas as pd
import os
import sqlite3
import json
import uuid
from streamlit_cookies_manager import EncryptedCookieManager
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import calculations as calc
import visualizations as viz

# ── DATABASE SETUP ──────────────────────
DB_PATH = "chats.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, data TEXT)")
    conn.commit()
    conn.close()

init_db()

# ── USER ID & CHAT ID ────────────────────
cookies = EncryptedCookieManager(
    prefix="rheoflow_",
    password=st.secrets.get("COOKIE_PASSWORD", "local-fallback-key-only")
)
if not cookies.ready():
    st.stop()  # Wait for cookies to load

if "user_id" not in cookies:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()

user_id = cookies["user_id"]

if "chat" not in st.query_params:
    st.query_params["chat"] = str(uuid.uuid4())
chat_id = st.query_params["chat"]

# ── SAVE FUNCTION ───────────────────────
def auto_save_chat():
    try:
        title = "New Conversation"
        if "chat_history" in st.session_state and st.session_state.chat_history:
            user_msgs = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
            if user_msgs:
                title = user_msgs[0][:22] + ("..." if len(user_msgs[0]) > 22 else "")
                
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        data_str = json.dumps({
            "samples": st.session_state.get("samples_list", []),
            "chat": st.session_state.get("chat_history", [])
        })
        cursor.execute("INSERT OR REPLACE INTO chats (id, user_id, title, data) VALUES (?, ?, ?, ?)", 
                       (chat_id, user_id, title, data_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Auto-save error: {e}")

# ── LOAD ACTIVE CHAT ─────────────────────
if "loaded_chat_id" not in st.session_state or st.session_state.loaded_chat_id != chat_id:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            saved_data = json.loads(row[0])
            st.session_state.samples_list = saved_data["samples"]
            st.session_state.chat_history = saved_data["chat"]
        else:
            raise Exception()
    except Exception:
        st.session_state.samples_list = [
            {"name": "Sample 1", "mw_val": 0.0, "mw_unit": "PPG",
             "t600": 0.0, "t300": 0.0, "g10s": 0.0, "g10m": 0.0}
        ]
        st.session_state.chat_history = []
    st.session_state.loaded_chat_id = chat_id

# ── PAGE CONFIG & STYLING ────────────────
st.set_page_config(page_title="RheoFlow AI - Multi-Sample Analyzer", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .gradient-text {
        background: linear-gradient(135deg, #FF007F 0%, #7F00FF 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    div[data-testid="stAlert"] {
        background-color: #12121A !important;
        color: #F1F5F9 !important;
        border: 1px solid #27273F !important;
        border-left: 5px solid #3B82F6 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.markdown("### 💬 RheoFlow AI Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.query_params["chat"] = str(uuid.uuid4())
        st.rerun()
    st.markdown("---")
    st.markdown("**Recent Conversations**")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM chats WHERE user_id = ?", (user_id,))
        past_chats = cursor.fetchall()
        conn.close()
    except Exception:
        past_chats = []

    if past_chats:
        for cid, title in past_chats:
            is_active = (cid == chat_id)
            label = f"👉 {title}" if is_active else f"💬 {title}"
            col_chat, col_del = st.columns([5, 1])
            with col_chat:
                if st.button(label, key=f"open_{cid}", use_container_width=True):
                    st.query_params["chat"] = cid
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{cid}"):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM chats WHERE id = ?", (cid,))
                    conn.commit()
                    conn.close()
                    if is_active:
                        st.query_params["chat"] = str(uuid.uuid4())
                    st.rerun()
    else:
        st.caption("No saved conversations yet.")

# ── MAIN HEADER ──────────────────────────
st.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-top: 10px; margin-bottom: 25px;">
    <span style="font-size: 2.2rem;">🧪</span>
    <h1 class="gradient-text" style="margin: 0; font-size: 2.2rem;">RheoFlow AI - Multi-Sample Drilling Fluid Analyzer</h1>
</div>
""", unsafe_allow_html=True)

# ── VECTOR DB CACHE ──────────────────────
@st.cache_resource
def load_vector_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception:
        return None

vector_db = load_vector_db()

GEMINI_API_KEYS = [
    st.secrets["GEMINI_API_KEY_1"],
    st.secrets["GEMINI_API_KEY_2"], 
    st.secrets["GEMINI_API_KEY_3"],
]

def build_full_context(results):
    context_lines = ["=== CALCULATED SAMPLE DATA ==="]
    for r in results:
        name = r.get("Sample Name", "Unknown")
        context_lines.append(f"\n--- {name} ---")
        context_lines.append(f"  Mud Weight (PPG): {r.get('MW (PPG)', 0):.3f}")
        context_lines.append(f"  Plastic Viscosity PV (cP): {r.get('PV (cP)', 0):.3f}")
        context_lines.append(f"  Yield Point YP (lb/100ft²): {r.get('YP (lb/100ft²)', 0):.3f}")
        context_lines.append(f"  Transport Index TI: {r.get('TI (YP/PV)', 0):.3f}")
        context_lines.append(f"  Flow Index n: {r.get('n (Flow Index)', 0):.3f}")
        context_lines.append(f"  Consistency Index k: {r.get('k (Consistency Index)', 0):.3f}")
    context_lines.append("\n=== END OF SAMPLE DATA ===")
    return "\n".join(context_lines)

def call_gemini_rag(query, full_context):
    local_context = ""
    if vector_db is not None:
        try:
            relevant_docs = vector_db.similarity_search(query, k=3)
            if relevant_docs:
                local_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        except Exception:
            pass

    rag_section = f"LOCAL STANDARDS:\n{local_context}" if local_context else "Use standard API RP 13B-1 rules."

    for i, api_key in enumerate(GEMINI_API_KEYS):
        try:
            key_client = genai.Client(api_key=api_key)
            prompt = f"{full_context}\n\n{rag_section}\n\nUSER QUESTION: {query}"
            response = key_client.models.generate_content(model="gemini-flash-latest", contents=prompt)
            return response.text, None
        except Exception as e:
            if i < len(GEMINI_API_KEYS) - 1:
                continue
            return f"Error: {str(e)}", None

# ── 1. SAMPLE MANAGER ────────────────────
st.subheader("1. Sample Manager")
col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    if st.button("➕ Add Another Sample", use_container_width=True):
        new_idx = len(st.session_state.samples_list) + 1
        st.session_state.samples_list.append({
            "name": f"Sample {new_idx}", "mw_val": 0.0, "mw_unit": "PPG", "t600": 0.0, "t300": 0.0, "g10s": 0.0, "g10m": 0.0
        })
        auto_save_chat()
        st.rerun()

with col_btn2:
    if st.button("🗑️ Remove Last Sample", use_container_width=True):
        if len(st.session_state.samples_list) > 1:
            st.session_state.samples_list.pop()
            auto_save_chat()
            st.rerun()
        else:
            st.warning("You must keep at least one sample.")

# ── 2. INPUT DATA ────────────────────────
st.subheader("2. Sample Input Data")
tabs = st.tabs([s["name"] for s in st.session_state.samples_list])

for idx, (tab, sample) in enumerate(zip(tabs, st.session_state.samples_list)):
    with tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            sample["name"] = st.text_input(f"Sample Name #{idx+1}", value=sample["name"], key=f"name_{idx}")
            sample["mw_val"] = st.number_input(f"Mud Weight Value", value=float(sample["mw_val"]), key=f"mw_val_{idx}")
            sample["mw_unit"] = st.selectbox(f"Mud Weight Unit", ["PPG", "SG", "lb/ft³", "kg/m³", "psi/1000ft"], 
                                             index=["PPG", "SG", "lb/ft³", "kg/m³", "psi/1000ft"].index(sample["mw_unit"]), key=f"mw_unit_{idx}")
        with col2:
            sample["t600"] = st.number_input(f"600 RPM Reading", value=float(sample["t600"]), key=f"t600_{idx}")
            sample["t300"] = st.number_input(f"300 RPM Reading", value=float(sample["t300"]), key=f"t300_{idx}")
        with col3:
            sample["g10s"] = st.number_input(f"10-Sec Gel", value=float(sample["g10s"]), key=f"g10s_{idx}")
            sample["g10m"] = st.number_input(f"10-Min Gel", value=float(sample["g10m"]), key=f"g10m_{idx}")

# ── CALCULATIONS & DISPLAY ───────────────
results = []
for sample in st.session_state.samples_list:
    res = calc.process_sample(
        name=sample["name"], mw_value=sample["mw_val"], mw_unit=sample["mw_unit"],
        theta600=sample["t600"], theta300=sample["t300"], gel10s=sample["g10s"], gel10m=sample["g10m"]
    )
    results.append(res)

df_results = pd.DataFrame(results)
full_data_context = build_full_context(results)

st.subheader("3. Multi-Sample Comparative Analysis Table")
df_display = df_results.set_index("Sample Name").T
st.dataframe(df_display.style.format("{:.3f}", na_rep="-"), use_container_width=True)

st.subheader("4. Rheological Visualizations")
viz_tab1, viz_tab2 = st.tabs(["📈 2D Flow Curves", "📊 Property Comparison"])
with viz_tab1:
    st.plotly_chart(viz.plot_2d_rheology_curves(results), use_container_width=True)
with viz_tab2:
    st.plotly_chart(viz.plot_comparative_bar_chart(results), use_container_width=True)

# ── CHATBOT ASSISTANT ───────────────────
st.subheader("⚡ RheoFlow AI Chatbot Assistant")

for msg in st.session_state.chat_history:
    avatar = "😊" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

if user_query := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    auto_save_chat() # Instant auto save
    
    with st.chat_message("user", avatar="😊"):
        st.write(user_query)

    with st.spinner("Analyzing..."):
        ai_reply, _ = call_gemini_rag(query=user_query, full_context=full_data_context)

    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
    auto_save_chat() # Instant auto save after AI reply
    
    with st.chat_message("assistant", avatar="✨"):
        st.write(ai_reply)
    
    st.rerun()

# Final state save on any UI parameter change
auto_save_chat()
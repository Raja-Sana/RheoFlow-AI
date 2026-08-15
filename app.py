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

# data save rakhny k liyeh helper functions
DB_PATH = "chats.db"
# Initialize database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, data TEXT)")
conn.commit()
conn.close()
# user ID (identifies by browser using streamlit cookie setting)
cookies = EncryptedCookieManager(
    prefix="rheoflow_",
    password=st.secrets["COOKIE_PASSWORD"]
)

if not cookies.ready():
    st.stop()  # Wait until cookies are loaded

if "uid" not in cookies:
    cookies["uid"] = str(uuid.uuid4())
    cookies.save()

user_id = cookies["uid"]
# Resolve active chat ID (identifies the current conversation)
if "chat" not in st.query_params:
    st.query_params["chat"] = str(uuid.uuid4())
chat_id = st.query_params["chat"]
# Load active chat data into session state
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
        # Default fresh state if chat doesn't exist
        st.session_state.samples_list = [
            {"name": "Sample 1", "mw_val": 0.0, "mw_unit": "PPG",
             "t600": 0.0, "t300": 0.0, "g10s": 0.0, "g10m": 0.0}
        ]
        st.session_state.chat_history = []
    st.session_state.loaded_chat_id = chat_id
# side bar ui
with st.sidebar:
    st.markdown("### 💬 RheoFlow AI Chat History")
    
    # Start a new conversation
    if st.button("➕ New Chat", use_container_width=True):
        st.query_params["chat"] = str(uuid.uuid4())
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Recent Conversations**")
    
    # Load all saved chats for the current user
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
            label = f"👉 {title} (Active)" if is_active else f"💬 {title}"
            
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
                    # If deleted chat was the active one, open a new chat
                    if is_active:
                        st.query_params["chat"] = str(uuid.uuid4())
                    st.rerun()
    else:
        st.caption("No saved conversations yet.")
# page ki configuration
st.set_page_config(page_title="RheoFlow AI - Multi-Sample Analyzer",page_icon="🧪",layout="wide")
st.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-top: 10px; margin-bottom: 25px;">
    <span style="font-size: 2.2rem; flex-shrink: 0; line-height: 1;">🧪</span>
    <h1 class="gradient-text" style="margin: 0; font-size: 2.2rem; line-height: 1.2;">RheoFlow AI - Multi-Sample Drilling Fluid Analyzer</h1>
</div>
""", unsafe_allow_html=True)

# custom styling k liyeh css
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* 1. Main Header Title (Emoji colored normally, text gradient applied) */
    .gradient-text {
        background: linear-gradient(135deg, #FF007F 0%, #7F00FF 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
        display: inline-block;
    }

    /* 2. Color-coded borders for different sections */
    /* Section 1: Sample Manager (Red Accent) */
    h3#1-sample-manager {
        border-left: 4px solid #EF4444;
        padding-left: 10px;
        color: #EF4444 !important;
    }
    
    /* Section 2: Input Data (Blue Accent) */
    h3#2-sample-input-data {
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
        color: #3B82F6 !important;
    }
    
    /* Section 3: Comparative Table (Green/Emerald Accent) */
    h3#3-multi-sample-comparative-analysis-table {
        border-left: 4px solid #10B981;
        padding-left: 10px;
        color: #10B981 !important;
    }
    
    /* Section 4: Visualizations (Purple Accent) */
    h3#4-rheological-visualizations-graphical-profiles {
        border-left: 4px solid #8B5CF6;
        padding-left: 10px;
        color: #8B5CF6 !important;
    }
    
    /* Section: Export & Share (Gold/Orange Accent) */
    h3#export-share-analysis {
        border-left: 4px solid #F59E0B;
        padding-left: 10px;
        color: #F59E0B !important;
    }
    
    /* Section: Chatbot (Pink/Rose Accent) */
    h3#rheoflow-ai-chatbot-assistant {
        border-left: 4px solid #EC4899;
        padding-left: 10px;
        color: #EC4899 !important;
    }

    /* 3. TRANSFORMS ST.INFO / ST.WARNING BOXES (Ugly blue to dark glassmorphic box) */
    div[data-testid="stAlert"] {
        background-color: #12121A !important;
        color: #F1F5F9 !important;
        border: 1px solid #27273F !important;
        border-left: 5px solid #3B82F6 !important; /* Left bar accent */
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
    }
    div[data-testid="stAlert"] p {
        color: #E2E8F0 !important;
        font-size: 14.5px !important;
        line-height: 1.6 !important;
    }

    /* 4. Styled Input Borders on focus */
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border: 1px solid #6366F1 !important;
        box-shadow: 0 0 8px rgba(99, 102, 241, 0.2) !important;
    }

    /* 5. Custom Styling for Tabs */
    button[data-baseweb="tab"] {
        font-size: 14.2px !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #E2E8F0 !important;
        border-bottom: 2px solid #6366F1 !important;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("Analyze, calculate, visualize, and evaluate rheological properties across multiple fluid formulations.")
with st.expander("📖 View API RP 13B-1 Rheology Formulas"):
    st.markdown("All calculations are based on the **API RP 13B-1** recommended practices for field testing of water-based drilling fluids:")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.latex(r"PV = \theta_{600} - \theta_{300} \quad \text{(Plastic Viscosity, cP)}")
        st.latex(r"AV = \frac{\theta_{600}}{2} \quad \text{(Apparent Viscosity, cP)}")
        st.latex(r"YP = \theta_{300} - PV \quad \text{(Yield Point, lb/100ft²)}")
        st.latex(r"TI = \frac{YP}{PV} \quad \text{(Transport Index)}")
    with col_f2:
        st.latex(r"n = 3.32 \log_{10}\left(\frac{\theta_{600}}{\theta_{300}}\right) \quad \text{(Flow Index)}")
        st.latex(r"k = \frac{\theta_{300}}{511^n} \quad \text{Consistency Index (lb/100ft}^2\text{·s}^n\text{)}")
        st.latex(r"BF = 1 - \frac{MW}{65.5} \quad \text{(Buoyancy Factor)}")

# multi api key setup agr eik ki limit khatm ho gi to dusri auto active ho jai gi
# Load API keys securely from Streamlit Secrets
GEMINI_API_KEYS = [
    st.secrets["GEMINI_API_KEY_1"],
    st.secrets["GEMINI_API_KEY_2"], 
    st.secrets["GEMINI_API_KEY_3"],
]

# session state ki initialization (with database auto-load fallback)

# loading faiss vector DB - eik bar load hoga jab app start hogi for fast cached data access
@st.cache_resource
def load_vector_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
        return db
    except Exception:
        return None  # agr faiss built nahi hoga abhi tk, chatbot gemini knowledge use kary ga 
vector_db = load_vector_db()

# ab ham chatbot k liyeh eik full context func genrate karian gy incluidng (numerical + graph coordinates) eik hi string mai taky gemini results genrate kary to uss k pass sari pic clear ho
def build_full_context(results):
    context_lines = []
    context_lines.append("=== CALCULATED SAMPLE DATA ===")

    for r in results:
        name = r.get("Sample Name", "Unknown")
        context_lines.append(f"\n--- {name} ---")
        context_lines.append(f"  Mud Weight (PPG)         : {r.get('MW (PPG)', 0):.3f}")
        context_lines.append(f"  Mud Weight (SG)          : {r.get('MW (SG)', 0):.3f}")
        context_lines.append(f"  Mud Weight (lb/ft³)      : {r.get('MW (lb/ft³)', 0):.3f}")
        context_lines.append(f"  Mud Weight (kg/m³)       : {r.get('MW (kg/m³)', 0):.3f}")
        context_lines.append(f"  Mud Weight (psi/1000ft)  : {r.get('MW (psi/1000ft)', 0):.3f}")
        context_lines.append(f"  Plastic Viscosity PV (cP): {r.get('PV (cP)', 0):.3f}")
        context_lines.append(f"  Apparent Viscosity AV(cP): {r.get('AV (cP)', 0):.3f}")
        context_lines.append(f"  Yield Point YP(lb/100ft²): {r.get('YP (lb/100ft²)', 0):.3f}")
        context_lines.append(f"  Transport Index TI       : {r.get('TI (YP/PV)', 0):.3f}")
        context_lines.append(f"  Flow Behavior Index n    : {r.get('n (Flow Index)', 0):.3f}")
        context_lines.append(f"  Consistency Index k      : {r.get('k (Consistency Index)', 0):.3f}")
        context_lines.append(f"  10-sec Gel (lb/100ft²)   : {r.get('Gel 10s (lb/100ft²)', 0):.3f}")
        context_lines.append(f"  10-min Gel (lb/100ft²)   : {r.get('Gel 10m (lb/100ft²)', 0):.3f}")
        context_lines.append(f"  Buoyancy Factor BF       : {r.get('Buoyancy Factor (BF)', 0):.3f}")

        # Graph coordinate data
        yp  = r.get('YP (lb/100ft²)', 0)
        t300 = r.get('300 RPM', 0)
        t600 = r.get('600 RPM', 0)
        context_lines.append(f"\n  2D Flow Curve Coordinates (Shear Rate → Shear Stress):")
        context_lines.append(f"    At Shear Rate 0    /s  → Shear Stress = {yp:.3f}  (= YP)")
        context_lines.append(f"    At Shear Rate 511  /s  → Shear Stress = {t300:.3f} (= θ300)")
        context_lines.append(f"    At Shear Rate 1022 /s  → Shear Stress = {t600:.3f} (= θ600)")

    context_lines.append("\n=== END OF SAMPLE DATA ===")
    return "\n".join(context_lines)

# rag aur gemini chatbot func development
def call_gemini_rag(query, full_context):
    """
    Full RAG pipeline:
    1. Search FAISS local database for relevant standards chunks.
    2. Build prompt with local context (if found) or Gemini fallback.
    3. Try each API key in order. If rate limited, try next key.
    """

    # rag step 1 Search local FAISS DB
    local_context = ""
    #rag_source_label = "Gemini AI knowledge (no local match found)"

    if vector_db is not None:
        try:
            relevant_docs = vector_db.similarity_search(query, k=3)
            if relevant_docs:
                local_context = "\n\n".join([doc.page_content for doc in relevant_docs])
                #rag_source_label = "Local FAISS standards database"
        except Exception:
            pass

    # rag step 2 Build RAG instruction 
    if local_context:
        rag_section = f"""
RETRIEVED FROM LOCAL STANDARDS DATABASE (Primary Source - high priority):
{local_context}

IMPORTANT: Base your answer PRIMARILY on the above retrieved standards.
Supplement with your engineering knowledge, but do NOT contradict these standards.
"""
    else:
        rag_section = """
No specific match found in the local standards database.
Use your comprehensive expert knowledge of API RP 13B-1 drilling fluid standards.
"""

    # rag step 3 Call Gemini with all context 
    for i, api_key in enumerate(GEMINI_API_KEYS):
        try:
            key_client = genai.Client(api_key=api_key)

            prompt = f"""
You are an expert AI Drilling Fluid Engineering Assistant with deep knowledge of API RP 13B-1.

{full_context}

{rag_section}

VISUALIZATION CONTEXT:
- "2D Flow Curves": Shear Rate (0, 511, 1022 /s) vs Shear Stress (YP, θ300, θ600).
  The steeper the curve slope, the higher the viscosity of the sample.
- "Property Comparison Bar Graph": Side-by-side bars showing PV, YP, TI for each sample.
  Taller PV bar = thicker mud. Taller YP bar = better suspension ability.

INSTRUCTIONS:
- Always refer to the SPECIFIC numerical values from the sample data above.
- If multiple samples exist, compare them DIRECTLY with exact numbers.
- If a value is outside the standard API range, clearly mention that values are not in API range.
- Describe graph shapes and trends based on the actual coordinate values.
- Keep your response professional, clear, and quantitative.

USER QUESTION: "{query}"
"""
            response = key_client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text, None

        except Exception as e:
            error_msg = str(e)
            # If rate-limited (429) OR Google servers are busy (503), try the next key!
            if any(term in error_msg.lower() for term in ["429", "503", "quota", "rate", "unavailable"]):
                if i < len(GEMINI_API_KEYS) - 1:
                    continue
                return f"Error: {error_msg}", None
            return f"Error: {error_msg}", None

    return "Error: Function exited without response.", None
# sample manager
st.subheader("1. Sample Manager")
col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    if st.button("➕ Add Another Sample", use_container_width=True):
        new_idx = len(st.session_state.samples_list) + 1
        st.session_state.samples_list.append({
            "name": f"Sample {new_idx}", "mw_val": 0, "mw_unit": "PPG", "t600": 0, "t300": 0, "g10s": 0, "g10m": 0
        })
        # Auto-save changes immediately after calculations run
       # save_user_session(user_id, st.session_state.samples_list, st.session_state.chat_history)
        st.rerun()

with col_btn2:
    if st.button("🗑️ Remove Last Sample", use_container_width=True):
        if len(st.session_state.samples_list) > 1:
            st.session_state.samples_list.pop()
            # Auto-save changes immediately after calculations run
           # save_user_session(user_id, st.session_state.samples_list, st.session_state.chat_history)
            st.rerun()
        else:
            st.warning("You must keep at least one sample.")
        

st.markdown("---")

# input data
st.subheader("2. Sample Input Data")
tabs = st.tabs([s["name"] for s in st.session_state.samples_list])

for idx, (tab, sample) in enumerate(zip(tabs, st.session_state.samples_list)):
    with tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### Sample Label & Density")
            sample["name"] = st.text_input(f"Sample Name #{idx+1}", value=sample["name"], key=f"name_{idx}")
            sample["mw_val"] = st.number_input(f"Mud Weight Value", value=float(sample["mw_val"]), min_value=0.0, step=1.0, key=f"mw_val_{idx}")
            sample["mw_unit"] = st.selectbox(f"Mud Weight Unit", ["PPG", "SG", "lb/ft³", "kg/m³", "psi/1000ft"], index=["PPG", "SG", "lb/ft³", "kg/m³", "psi/1000ft"].index(sample["mw_unit"]), key=f"mw_unit_{idx}")
        with col2:
            st.markdown("##### Viscometer Dial Readings")
            sample["t600"] = st.number_input(f"600 RPM Reading (\u03b8\u2086\u2080\u2080)", value=float(sample["t600"]), min_value=0.0, step=5.0, key=f"t600_{idx}")
            sample["t300"] = st.number_input(f"300 RPM Reading (\u03b8\u2083\u2080\u2080)", value=float(sample["t300"]), min_value=0.0, step=5.0, key=f"t300_{idx}")
        with col3:
            st.markdown("##### Gel Strengths")
            sample["g10s"] = st.number_input(f"10-Sec Gel (lb/100ft²)", value=float(sample["g10s"]), min_value=0.0, step=1.0, key=f"g10s_{idx}")
            sample["g10m"] = st.number_input(f"10-Min Gel (lb/100ft²)", value=float(sample["g10m"]), min_value=0.0, step=1.0, key=f"g10m_{idx}")
st.markdown("---")

# process calculations
results = []
for sample in st.session_state.samples_list:
    res = calc.process_sample(
        name=sample["name"],
        mw_value=sample["mw_val"],
        mw_unit=sample["mw_unit"],
        theta600=sample["t600"],
        theta300=sample["t300"],
        gel10s=sample["g10s"],
        gel10m=sample["g10m"]
    )
    results.append(res)
    # -----------------------------------------------
    # VALIDATIONS (Shows warnings tagged with sample name)
    # -----------------------------------------------
    rheo_warning = calc.validate_rheology(sample["t600"], sample["t300"])
    if rheo_warning:
        st.warning(f"⚠️ **[{sample['name']}]** {rheo_warning}")
        
    yp_warning = calc.validate_yp(res["YP (lb/100ft²)"])
    if yp_warning:
        st.warning(f"⚠️ **[{sample['name']}]** {yp_warning}")
        
    bf_warning = calc.validate_bf(res["Buoyancy Factor (BF)"])
    if bf_warning:
        st.warning(f"⚠️ **[{sample['name']}]** {bf_warning}")
df_results = pd.DataFrame(results)

# Building full context once after calculations this will automatically available to the chatbot every time
full_data_context = build_full_context(results)
# Auto-save changes immediately after calculations run
#save_user_session(user_id, st.session_state.samples_list, st.session_state.chat_history)


# comparative table
# -----------------------------------------------
st.subheader("3. Multi-Sample Comparative Analysis Table")
df_display = df_results.set_index("Sample Name").T
st.dataframe(df_display.style.format("{:.3f}", na_rep="-"), use_container_width=True)
st.markdown("---")

# visualizations
st.subheader("4. Rheological Visualizations & Graphical Profiles")
viz_tab1, viz_tab2 = st.tabs(["📈 2D Flow Curves", "📊 Property Comparison"])

with viz_tab1:
    st.plotly_chart(viz.plot_2d_rheology_curves(results), use_container_width=True, config={'displayModeBar': False})
with viz_tab2:
    st.plotly_chart(viz.plot_comparative_bar_chart(results), use_container_width=True, config={'displayModeBar': False})
st.markdown("---")

st.markdown("---")

# AI CHATBOT 
st.subheader("⚡ RheoFlow AI Chatbot Assistant")
#st.caption("Powered by Google Gemini • Local FAISS RAG → Gemini Fallback • Full sample context")

# showing RAG status
if vector_db is not None:
    st.success("✅ Local FAISS standards database loaded. Chatbot will search locally first.")
else:
    st.warning("⚠️ No FAISS database found. Run create_vector_db.py first. Using Gemini knowledge only.")

# Render chat history
for msg in st.session_state.chat_history:
    avatar = "😊" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Chat input
if user_query := st.chat_input("Ask a question (e.g., 'Compare my PV and YP values with API standards')..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    # Auto-save changes immediately after calculations run
   # save_user_session(user_id, st.session_state.samples_list, st.session_state.chat_history)
    with st.chat_message("user", avatar="😊"):
        st.write(user_query)

    with st.spinner("🔬 Searching local database then querying Gemini..."):
        ai_reply, pdf_url = call_gemini_rag(
            query=user_query,
            full_context=full_data_context
        )

    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
    # Auto-save changes immediately after calculations run
    #save_user_session(user_id, st.session_state.samples_list, st.session_state.chat_history)
    with st.chat_message("assistant", avatar="✨"):
        st.write(ai_reply)

# auto save logic code

def auto_save_chat():
    try:
        # Determine a summary title from the first question
        title = "New Conversation"
        if st.session_state.chat_history:
            user_msgs = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
            if user_msgs:
                title = user_msgs[0][:22] + ("..." if len(user_msgs[0]) > 22 else "")
                
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        data_str = json.dumps({
            "samples": st.session_state.samples_list,
            "chat": st.session_state.chat_history
        })
        cursor.execute("INSERT OR REPLACE INTO chats (id, user_id, title, data) VALUES (?, ?, ?, ?)", (chat_id, user_id, title, data_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Auto-save error: {e}")

auto_save_chat()
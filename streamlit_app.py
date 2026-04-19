import streamlit as st
import os
import uuid
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv

# Langchain / AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Local Modules
from rag_engine import RAGEngine

load_dotenv()

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="NOVA AI Tutor", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    /* Neon Dark Theme */
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    /* Title Styling */
    .nova-title {
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 3rem;
        background: linear-gradient(45deg, #00ffcc, #00b3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 2rem;
    }
    
    /* User Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="block"] {
        background-color: #141414;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #333;
    }
    
    /* AI Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
        border-left: 3px solid #00ffcc;
        background: linear-gradient(90deg, rgba(0,255,204,0.05) 0%, transparent 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='nova-title'>NOVA AI TUTOR</div>", unsafe_allow_html=True)

# ==========================================
# INITIALIZATION
# ==========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or "your_gemini_api_key_here" in GEMINI_API_KEY:
    st.warning("⚠️ Please set your GEMINI_API_KEY in the .env file.")
    st.stop()

client = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def optimize_image(img_file):
    """Resizes and compresses image for API payload without cropping."""
    img = Image.open(img_file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Maintain aspect ratio, max dimension 1024
    img.thumbnail((1024, 1024), Image.LANCZOS)
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    optimized_b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{optimized_b64}"

# ==========================================
# SIDEBAR (Vision Center)
# ==========================================
with st.sidebar:
    st.header("👁️ Vision Center")
    st.markdown("Upload a math problem or diagram for Nova to analyze.")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Current Visual Data", use_column_width=True)
        st.success("Visual Data Locked In")
        
    st.markdown("---")
    if st.button("🧹 Clear Memory", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# CHAT INTERFACE
# ==========================================
# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
prompt = st.chat_input("Ask Nova a question...")

if prompt:
    # 1. Show User Input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process Image (if any)
    active_image = optimize_image(uploaded_file) if uploaded_file else None

    # 3. Retrieve Context from RAG
    clean_text = prompt.lower().strip()
    is_greeting = any(clean_text.startswith(g) for g in ["hi", "hello", "hey", "yo"]) and len(clean_text) < 10
    
    semantic_context = ""
    if not is_greeting:
        needs_history = any(w in clean_text for w in ["remember", "previous", "earlier", "told", "said"])
        needs_docs = any(w in clean_text for w in ["document", "pdf", "data", "file"])
        
        context_parts = []
        if needs_history:
            relevant_history = st.session_state.rag.query_relevant_history(prompt, limit=5)
            if relevant_history:
                context_parts.append("\n".join([f"Past conversation: {doc.page_content}" for doc in relevant_history]))
        if needs_docs:
            relevant_docs = st.session_state.rag.query_documents(prompt, limit=5)
            if relevant_docs:
                context_parts.append("\n".join([f"Document excerpt: {doc.page_content}" for doc in relevant_docs]))
        if context_parts:
            semantic_context = "\n\n".join(context_parts)

    # 4. Construct System Prompt
    system_content = (
        "You are Nova, an Elite AI Reasoning Tutor. "
        "CRITICAL: You are a native multi-modal AI. You can see images directly. "
        "If an image is attached to the user's query, analyze it meticulously. "
        "Never claim you cannot see. Format output beautifully using Markdown."
    )
    if semantic_context:
        system_content += f"\n\n[RELEVANT PAST CONTEXT]:\n{semantic_context}"

    # 5. Build Final Message Chain
    langchain_msgs = [SystemMessage(content=system_content)]
    
    # Add previous chat history (as text context for Gemini)
    for m in st.session_state.messages[:-1]: 
        if m["role"] == "user":
            langchain_msgs.append(HumanMessage(content=m["content"]))
        else:
            langchain_msgs.append(AIMessage(content=m["content"]))
            
    # Add current query with Multi-Modal Image
    if active_image:
        langchain_msgs.append(HumanMessage(content=[
            {"type": "text", "text": f"STUDENT_QUERY: {prompt}"},
            {"type": "image_url", "image_url": {"url": active_image}}
        ]))
    else:
        langchain_msgs.append(HumanMessage(content=f"STUDENT_QUERY: {prompt}"))

    # 6. Stream Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Synchronous streaming loop for Streamlit
        for chunk in client.stream(langchain_msgs):
            if chunk.content:
                full_response += chunk.content
                # Update placeholder with a blinking cursor
                message_placeholder.markdown(full_response + "▌")
                
        # Final render without cursor
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})

import streamlit as st
import os
import uuid
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# NEW: We deleted Google and imported Groq's ultra-fast brain!
from langchain_groq import ChatGroq

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="CivicSight", 
    page_icon="🏛️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 CLEAN CSS
# ==========================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stSidebar"] {
        background-color: #f0f4f9; 
    }

    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #202124 !important;
    }
    
    button[kind="primary"] {
        background-color: #1a73e8 !important; 
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
    }
    button[kind="primary"] p {
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #1557b0 !important;
    }

    div[data-testid="stSidebar"] button[kind="secondary"] {
        border: none !important;
        background-color: transparent !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #e4e9f1 !important;
    }

    [data-testid="stPopover"] button svg,
    [data-testid="stPopover"] button div[data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stPopover"] button {
        border: none !important;
        background-color: transparent !important;
        padding: 0px !important;
        width: 30px !important;
    }
    
    [data-testid="stChatInput"] {
        border: 1px solid #dadce0 !important;
        border-radius: 24px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        margin-bottom: 20px !important;
        transform: scale(1.10) !important; 
        transform-origin: bottom center !important;
    }
    textarea {
        font-size: 16px !important;
    }
    
    .welcome-title {
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #202124;
        margin-top: 15vh;
        margin-bottom: 20px;
    }
    .stMarkdown p {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 AI AGENT (LANGGRAPH + GROQ)
# ==========================================
@st.cache_resource
def load_rag_pipeline():
    load_dotenv()
    
    # NEW: We are now using Meta's Llama-3.1 model on Groq's lightning-fast servers!
    llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)
    
    # This runs locally on your CPU, so it stays exactly the same!
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vector_db = Chroma(persist_directory=os.path.join(current_dir, "../vector_db"), embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    @tool
    def local_civic_database(query: str) -> str:
        """Search for information about local budgets, smart city plans, and committee decisions in local PDF files."""
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    
    @tool
    def web_search_tool(query: str) -> str:
        """Search the internet for civic information."""
        try:
            results = DDGS().text(query, max_results=3)
            return "\n\n".join([f"Source: {r['href']}\nInfo: {r['body']}" for r in results])
        except Exception:
            return "Web search failed. Could not find information."
    
    tools = [local_civic_database, web_search_tool]
    
    agent_executor = create_react_agent(llm, tools)
    
    return agent_executor

rag_chain = load_rag_pipeline()

# ==========================================
# 💾 STATE MANAGEMENT
# ==========================================
if "chats" not in st.session_state:
    default_id = str(uuid.uuid4())
    st.session_state.chats = {default_id: {"title": "New Chat", "messages": [], "pinned": False}}
    st.session_state.current_chat_id = default_id

current_id = st.session_state.current_chat_id

# ==========================================
# 📂 SIDEBAR
# ==========================================
with st.sidebar:
    
    if st.button("📝 New chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": [], "pinned": False}
        st.session_state.current_chat_id = new_id
        st.rerun()
        
    pinned_chats = {k: v for k, v in st.session_state.chats.items() if v.get("pinned", False) and len(v["messages"]) > 0}
    if pinned_chats:
        st.markdown("<br><p style='font-size: 13px !important; color: #5f6368 !important; font-weight: bold !important;'>Pinned</p>", unsafe_allow_html=True)
        for chat_id, chat_data in pinned_chats.items():
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"💬 {chat_data['title']}", key=f"pin_btn_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with col2:
                with st.popover("⋮"):
                    if st.button("Unpin", key=f"unpin_{chat_id}"):
                        st.session_state.chats[chat_id]["pinned"] = False
                        st.rerun()
                    if st.button("🗑️ Delete", key=f"del1_{chat_id}"):
                        del st.session_state.chats[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0] if st.session_state.chats else str(uuid.uuid4())
                        st.rerun()

    st.markdown("<br><p style='font-size: 13px !important; color: #5f6368 !important; font-weight: bold !important;'>Recents</p>", unsafe_allow_html=True)
    
    for chat_id, chat_data in list(st.session_state.chats.items()):
        if chat_data.get("pinned", False) or (len(chat_data["messages"]) == 0 and chat_id != current_id):
            continue 
            
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"💬 {chat_data['title']}", key=f"rec_btn_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            with st.popover("⋮"):
                new_title = st.text_input("Rename", value=chat_data["title"], key=f"ren_{chat_id}")
                if st.button("Save", key=f"save_{chat_id}"):
                    st.session_state.chats[chat_id]["title"] = new_title
                    st.rerun()
                if st.button("📌 Pin Chat", key=f"pin_{chat_id}"):
                    st.session_state.chats[chat_id]["pinned"] = True
                    st.rerun()
                if st.button("🗑️ Delete", key=f"del2_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        if len(st.session_state.chats) > 0:
                            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                        else:
                            new_id = str(uuid.uuid4())
                            st.session_state.chats[new_id] = {"title": "New Chat", "messages": [], "pinned": False}
                            st.session_state.current_chat_id = new_id
                    st.rerun()

# ==========================================
# 🖥️ MAIN UI
# ==========================================

is_empty = len(st.session_state.chats[current_id]["messages"]) == 0
if is_empty:
    st.markdown('<h1 class="welcome-title">What do you want to know today?</h1>', unsafe_allow_html=True)

for message in st.session_state.chats[current_id]["messages"]:
    avatar = "🧑‍💻" if message["role"] == "user" else "🏛️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask anything..."):
    
    if is_empty:
        st.session_state.chats[current_id]["title"] = user_question[:25] + "..."

    st.session_state.chats[current_id]["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar="🏛️"):
        
        status_box = st.empty()
        
        with status_box.container():
            with st.spinner("Searching records and the web..."):
                
                system_instruction = """You are CivicSight, a strict and highly intelligent Civic Assistant.
                CRITICAL RULES YOU MUST FOLLOW:
                1. You are RESTRICTED to answering ONLY questions related to civics, local politics, city development, government policies, infrastructure, and public administration.
                2. If the user asks ANY question unrelated to the topics above (e.g., movies, sports, coding, general trivia, weather, cooking, personal advice), you MUST refuse to answer. You MUST reply EXACTLY with: "I am restricted to answering only politics and city development-related questions."
                3. Always check the 'local_civic_database' tool first for answers.
                4. If the answer is not in the database, use the web search tool, but ALWAYS add 'Pune municipal corporation' to your search query.
                """
                
                answer = "An error occurred."
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        response = rag_chain.invoke({
                            "messages": [
                                ("system", system_instruction),
                                ("user", user_question)
                            ]
                        })
                        
                        raw_answer = response["messages"][-1].content
                        if isinstance(raw_answer, list):
                            answer = "".join(block["text"] for block in raw_answer if block.get("type") == "text")
                        else:
                            answer = str(raw_answer)
                            
                        break 
                        
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            if attempt < max_retries - 1:
                                st.warning(f"⚠️ API Speed Limit Reached. Auto-cooling down for 60 seconds... (Attempt {attempt+1} of {max_retries})")
                                time.sleep(60)
                            else:
                                answer = "Error: API limit completely exhausted. Please try again later."
                                break
                        else:
                            answer = f"Error: {str(e)}"
                            break
                            
        status_box.empty()
        st.markdown(answer)
            
    st.session_state.chats[current_id]["messages"].append({"role": "assistant", "content": answer})
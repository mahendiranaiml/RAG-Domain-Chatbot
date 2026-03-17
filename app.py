import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from src.ingest import embedde
from src.retriever import chunk_retrieve
from src.generator import generate

load_dotenv()

# --- 0. CREATE LLM ONCE ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ")
)

# --- 1. RESOURCE LOADING (Safely) ---
@st.cache_resource 
def load_resources():
    # If folder doesn't exist, return None to avoid the 'No such file' error
    if not os.path.exists("vectorstore"):
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            "vectorstore", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception:
        return None

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="PDF AI Assistant", layout="wide")
st.title("🚀 Advanced RAG Chatbot")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Process & Ingest"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                # Ensure docs folder exists
                if not os.path.exists("docs"):
                    os.makedirs("docs")
                    
                file_path = os.path.join("docs", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create the vectorstore and load chunks and docs
                
                chunks, docs = embedde(file_path)
                
                # Persist chunks and docs in session_state so they survive reruns
                st.session_state["chunks"] = chunks
                st.session_state["docs"] = docs
                
                st.success("Ingestion complete!")
                st.cache_resource.clear() # Reset cache to load new data
                st.rerun() # Refresh the app
        else:
            st.error("Please upload a file.")

# --- 3. CHAT LOGIC ---
vectorstore = load_resources()

if vectorstore is None:
    st.info("👈 Please upload a PDF and click 'Process' in the sidebar to start.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if query := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Analyzing..."):
            # 1. RETRIEVE — use docs and chunks from session_state
            docs = st.session_state.get("docs", [])
            chunks = st.session_state.get("chunks", [])
            final_pages_for_llm = chunk_retrieve(query, docs, vectorstore, chunks, llm)

            # 2. GENERATE
            with st.chat_message("assistant"):
                response = generate(query, llm, final_pages_for_llm)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
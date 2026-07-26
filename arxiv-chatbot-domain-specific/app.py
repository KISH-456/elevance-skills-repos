import streamlit as st
import pandas as pd
from rag_backend import ArxivRAG
import os

# Page configuration
st.set_page_config(page_title="ArXiv CS Expert Chatbot", layout="wide")

# Initialize Backend
@st.cache_resource
def get_rag():
    rag = ArxivRAG("arxiv_cs.jsonl")
    if not os.path.exists("faiss_index"):
        rag.load_and_index(limit=2000)
    return rag

rag = get_rag()

# Sidebar for paper search
st.sidebar.title("🔍 Paper Search")
search_query = st.sidebar.text_input("Search for papers (e.g., 'Transformer')")
if search_query:
    results = rag.search(search_query, k=5)
    for res in results:
        with st.sidebar.expander(res.metadata['title']):
            st.write(f"**Authors:** {res.metadata['authors']}")
            st.write(f"**Abstract:** {res.page_content}")
            st.write(f"**ID:** {res.metadata['id']}")

# Main Chat Interface
st.title("🎓 ArXiv CS Expert Chatbot")
st.markdown("Discuss advanced Computer Science topics, summarize papers, and visualize concepts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about CS research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.get_answer(prompt)
            answer = response['result']
            sources = response['source_documents']
            
            st.markdown(answer)
            if sources:
                with st.expander("📚 View Sources"):
                    for doc in sources:
                        st.write(f"- **{doc.metadata['title']}** (ID: {doc.metadata['id']})")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Concept Visualization (Placeholder for D2/Mermaid)
st.sidebar.markdown("---")
st.sidebar.title("📊 Concept Visualization")
viz_concept = st.sidebar.text_input("Enter concept to visualize")
if viz_concept:
    st.sidebar.info(f"Visualizing '{viz_concept}'... (This feature uses LLM to generate Mermaid diagrams)")
    # Logic to generate diagram via LLM could be added here
    st.sidebar.code("""
    graph TD
    A[Concept] --> B[Sub-concept 1]
    A --> C[Sub-concept 2]
    """, language="mermaid")

"""
Interactive Streamlit interface for DocQuery RAG system.

Run with: streamlit run streamlit_app/app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --- Page Config ---
st.set_page_config(
    page_title="DocQuery - Chat with Your Documents",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocQuery")
st.markdown("Upload PDFs. Ask questions. Get cited answers.")


# --- Initialize Pipeline ---
@st.cache_resource
def get_pipeline():
    from rag_pipeline import RAGPipeline
    return RAGPipeline()


pipeline = get_pipeline()


# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if already ingested
            existing_docs = pipeline.get_documents()
            doc_name = uploaded_file.name.replace(".pdf", "")

            if doc_name not in existing_docs:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    stats = pipeline.ingest_uploaded_pdf(
                        uploaded_file.read(),
                        uploaded_file.name,
                    )
                    st.success(
                        f"✅ {uploaded_file.name}\n"
                        f"{stats['total_pages']} pages → {stats['num_chunks']} chunks"
                    )

    # Show ingested documents
    docs = pipeline.get_documents()
    if docs:
        st.markdown("**Loaded documents:**")
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}"):
                pipeline.remove_document(doc)
                st.rerun()

        if st.button("Clear All Documents"):
            pipeline.clear_all()
            st.rerun()
    else:
        st.info("Upload a PDF to get started")

    st.markdown("---")

    # Filter by document
    filter_doc = None
    if len(docs) > 1:
        filter_option = st.selectbox(
            "Search in:",
            ["All documents"] + docs,
        )
        if filter_option != "All documents":
            filter_doc = filter_option

    # Settings
    with st.expander("⚙️ Settings"):
        show_sources = st.checkbox("Show source passages", value=True)
        show_scores = st.checkbox("Show relevance scores", value=False)


# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            if show_sources and message["sources"]:
                with st.expander(f"📚 Sources ({len(message['sources'])} passages)"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(
                            f"**[Source {i+1}]** {source['document']}, "
                            f"Page {source['page']}"
                        )
                        if show_scores:
                            st.caption(f"Relevance: {source.get('relevance_score', 0):.3f}")
                        st.markdown(f"> {source['text_preview']}")
                        st.markdown("---")


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not docs:
        st.warning("Please upload a document first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = pipeline.query(prompt, filter_doc=filter_doc)

            st.markdown(result["answer"])

            # Show sources
            if show_sources and result["sources"]:
                with st.expander(f"📚 Sources ({len(result['sources'])} passages)"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(
                            f"**[Source {i+1}]** {source['document']}, "
                            f"Page {source['page']}"
                        )
                        if show_scores:
                            st.caption(f"Relevance: {source.get('relevance_score', 0):.3f}")
                        st.markdown(f"> {source['text_preview']}")
                        st.markdown("---")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })


# --- Footer ---
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.markdown("Built by **Ayoob Amaar**")
col2.markdown(f"📄 {len(docs)} document(s) loaded")
col3.markdown(f"💬 {len(st.session_state.messages) // 2} questions asked")

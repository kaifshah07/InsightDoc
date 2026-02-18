# import streamlit as st
# from rag_engine import load_index, retrieve_chunks, build_context, generate_answers

# # Page config
# st.set_page_config(
#     page_title="InsightDoc - Document QA",
#     page_icon="📄",
#     layout="wide"
# )

# st.title("📄 InsightDoc - Document Question Answering System")
# st.write("Ask questions from your uploaded PDF documents.")

# # Load index (only once)
# @st.cache_resource
# def load_rag_components():
#     vectorizer, tfidf_matrix, all_chunks = load_index()
#     return vectorizer, tfidf_matrix, all_chunks

# try:
#     vectorizer, tfidf_matrix, all_chunks = load_rag_components()
# except:
#     st.error("Index not found. Please run build_index.py first.")
#     st.stop()

# # User input
# query = st.text_input("Enter your question:")

# if st.button("Ask"):

#     if not query.strip():
#         st.warning("Please enter a valid question.")
#         st.stop()

#     with st.spinner("Searching documents..."):

#         indices, similarity_scores = retrieve_chunks(
#             query,
#             vectorizer,
#             tfidf_matrix
#         )

#         if not indices:
#             st.warning("No relevant information found.")
#             st.stop()

#         context = build_context(
#             indices,
#             all_chunks,
#             similarity_scores
#         )

#         answer = generate_answers(query, context)

#     st.subheader("Answer:")
#     st.write(answer)

#     st.subheader("Sources Used:")
#     for rank, idx in enumerate(indices, start=1):
#         chunk = all_chunks[idx]
#         st.write(
#             f"Source {rank}: {chunk['source']} "
#             f"(Chunk {chunk['chunk_id']})"
#         )

import streamlit as st
from rag_engine import load_index, retrieve_chunks, build_context, generate_answers

# --------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------
st.set_page_config(
    page_title="InsightDoc - RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------
# TITLE SECTION
# --------------------------------------------
st.title("📄 InsightDoc - Document Question Answering System")
st.markdown("Ask questions from your indexed PDF documents using RAG + Local LLM.")

st.divider()

# --------------------------------------------
# LOAD INDEX (cached)
# --------------------------------------------
@st.cache_resource
def load_resources():
    return load_index()

vectorizer, tfidf_matrix, all_chunks = load_resources()

# --------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=10,
    value=5
)

min_similarity = st.sidebar.slider(
    "Minimum similarity threshold",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.01
)

model_name = st.sidebar.text_input(
    "LLM Model Name (Ollama)",
    value="qwen:1.8b"
)

st.sidebar.markdown("---")
st.sidebar.info("Make sure Ollama is running locally.")

# --------------------------------------------
# QUERY INPUT
# --------------------------------------------
query = st.text_input("🔎 Enter your question:")

if query:

    with st.spinner("🔍 Retrieving relevant chunks..."):

        indices, similarity_scores = retrieve_chunks(
            query,
            vectorizer,
            tfidf_matrix,
            min_similarity=min_similarity,
            top_k=top_k
        )

        if not indices:
            st.warning("No relevant chunks found.")
            st.stop()

        context = build_context(
            indices,
            all_chunks,
            similarity_scores
        )

    with st.spinner("🤖 Generating answer from LLM..."):

        answer = generate_answers(
            query,
            context,
            model=model_name
        )

    # --------------------------------------------
    # ANSWER DISPLAY
    # --------------------------------------------
    st.subheader("📌 Answer")
    st.success(answer)

    # --------------------------------------------
    # SOURCE DISPLAY
    # --------------------------------------------
    with st.expander("📚 Sources Used"):

        for rank, idx in enumerate(indices, start=1):
            chunk = all_chunks[idx]

            st.markdown(f"""
            **Source {rank}**
            - PDF: `{chunk['source']}`
            - Chunk ID: `{chunk['chunk_id']}`
            - Similarity: `{similarity_scores[idx]:.3f}`
            """)

            st.write(chunk["text"])
            st.divider()

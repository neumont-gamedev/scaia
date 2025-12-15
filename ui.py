import sys
import streamlit as st

st.write("Python executable:", sys.executable)

import os
from rag import load_qa_chain

st.set_page_config(
    page_title="Course Assistant",
    layout="centered"
)

st.title("Course Assistant")
st.caption("Answers are generated from course materials and include citations.")

if "qa" not in st.session_state:
    st.session_state.qa = load_qa_chain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about the course material")

if st.button("Ask") and question:
    result = st.session_state.qa.invoke({"query": question})

    answer = result["result"]
    sources = result["source_documents"]

    st.session_state.history.append((question, answer, sources))

for q, a, srcs in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")

    if srcs:
        with st.expander("Sources"):
            seen = set()
            for doc in srcs:
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page")
                label = f"{source}, page {page}" if page else source
                if label not in seen:
                    st.markdown(f"- {label}")
                    seen.add(label)

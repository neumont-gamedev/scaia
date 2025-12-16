import os
import streamlit as st
from rag import load_qa_chain

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Course Assistant",
    layout="centered"
)

# ---------------- Light sidebar styling ----------------
st.markdown("""
<style>
    /* Sidebar buttons (question history) */
    section[data-testid="stSidebar"] button {
        width: 100%;
        text-align: left;
        padding: 0.25rem 0.5rem;
        margin-bottom: 0.25rem;
    }

    /* Sidebar header layout */
    .sidebar-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.75rem;
        margin-bottom: 0.25rem;
    }

    .sidebar-subtitle {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Helper functions ----------------
def question_preview(text, max_len=45):
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_len else text[:max_len].rstrip() + "…"

def get_mode_instruction(mode: str) -> str:
    if mode == "Hint":
        return (
            "Give a short hint that helps the student think, "
            "but do not provide a full answer. Ask a guiding question."
        )

    if mode == "Exam Prep":
        return (
            "Answer concisely and precisely. Use correct terminology. "
            "Structure the response as bullet points when appropriate."
        )

    # Default: Explain
    return (
        "Provide a clear, instructor-style explanation suitable "
        "for a college-level course."
    )


# ---------------- Session state initialization ----------------
if "qa" not in st.session_state:
    st.session_state.qa = load_qa_chain()

if "history" not in st.session_state:
    # Each entry: (question, answer, sources)
    st.session_state.history = []

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

# ---------------- Sidebar ----------------
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/n-neumont.png", width=100)

    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">SCAIA</div>
        <div class="sidebar-subtitle">Student Course AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Ask questions about course materials.\n\n"
        "Answers are generated from course documents and include citations."
    )

    st.markdown("### Response Mode")

    prompt_mode = st.radio(
        label="",
        options=[
            "Explain",
            "Hint",
            "Exam Prep"
        ],
        index=0
    )

    st.caption(
        "Controls how detailed the assistant's response is."
    )

    st.session_state.prompt_mode = prompt_mode

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.conversation_summary = ""
        st.session_state.selected_index = None        

    st.markdown("### Question History")

    if not st.session_state.history:
        st.caption("No questions yet.")
    else:
        for i, (q, _, _) in enumerate(st.session_state.history):
            label = question_preview(q)

            if st.session_state.selected_index == i:
                st.markdown(f"👉 **{label}**")
            else:
                if st.button(label, key=f"q_{i}"):
                    st.session_state.selected_index = i

# ---------------- Form ----------------
with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Ask a question")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    mode_instruction = get_mode_instruction(
        st.session_state.prompt_mode
    )

    augmented_question = (
        f"Response mode instruction:\n"
        f"{mode_instruction}\n\n"
        f"Conversation so far:\n"
        f"{st.session_state.conversation_summary}\n\n"
        f"Student question:\n"
        f"{question}"
    )

    try:
        with st.spinner("Thinking…"):
            result = st.session_state.qa.invoke({
                "query": augmented_question
            })
    except Exception as e:
        st.error(f"Error during QA invocation: {e}")
        st.stop()


    answer = result["result"]
    sources = result["source_documents"]

    # Update conversation memory
    st.session_state.conversation_summary += (
        f"User asked: {question}\n"
        f"Assistant answered: {answer}\n"
    )

    MAX_CHARS = 1500
    if len(st.session_state.conversation_summary) > MAX_CHARS:
        st.session_state.conversation_summary = (
            st.session_state.conversation_summary[-MAX_CHARS:]
        )

    st.session_state.history.append((question, answer, sources))
    st.session_state.selected_index = len(st.session_state.history) - 1
    st.rerun()
st.divider()

# ---------------- Display conversation ----------------
if st.session_state.selected_index is not None:
    # Show selected question from sidebar
    q, a, srcs = st.session_state.history[
        st.session_state.selected_index
    ]

    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")

    if srcs:
        with st.expander("Sources"):
            seen = set()
            for doc in srcs:
                source = os.path.basename(
                    doc.metadata.get("source", "unknown")
                )
                page = doc.metadata.get("page")
                label = f"{source}, page {page}" if page else source

                if label not in seen:
                    st.markdown(f"- {label}")
                    seen.add(label)

else:
    # Default view: show most recent first
    for q, a, srcs in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

        if srcs:
            with st.expander("Sources"):
                seen = set()
                for doc in srcs:
                    source = os.path.basename(
                        doc.metadata.get("source", "unknown")
                    )
                    page = doc.metadata.get("page")
                    label = f"{source}, page {page}" if page else source

                    if label not in seen:
                        st.markdown(f"- {label}")
                        seen.add(label)

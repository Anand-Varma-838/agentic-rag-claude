"""
app.py
------
Streamlit chat interface for the Agentic RAG system.

Run: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import AgentRunner
from retriever import HybridRetriever
from vectorstore import VectorStore
from tracer import AgentTracer

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Agentic RAG")
st.caption("Powered by Claude · Hybrid retrieval · Multi-hop reasoning · Source citations")

# ---------------------------------------------------------------------------
# Session state — initialise once per session
# ---------------------------------------------------------------------------

if "ready" not in st.session_state:
    with st.spinner("Loading models (first run may take ~30s to download)..."):
        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store=vector_store)
        # Build BM25 index from whatever is already stored
        retriever.build_bm25_index()
        tracer = AgentTracer()
        st.session_state.agent = AgentRunner(retriever=retriever, tracer=tracer)
        st.session_state.tracer = tracer
        st.session_state.messages = []   # list of {role, content}
        st.session_state.traces = []     # parallel list of trace snapshots
        st.session_state.ready = True

agent: AgentRunner = st.session_state.agent
tracer: AgentTracer = st.session_state.tracer

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📂 Document Store")

    doc_count = agent.retriever.vector_store.count()
    st.metric("Chunks indexed", doc_count)

    if doc_count == 0:
        st.warning(
            "No documents indexed yet.\n\n"
            "Drop files into `data/docs/` then run:\n"
            "```\npython ingest.py\n```"
        )
    else:
        st.success(f"{doc_count} chunks ready to query")

    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        agent.reset()
        st.session_state.messages = []
        st.session_state.traces = []
        st.rerun()

    st.divider()
    st.caption("**Try asking:**")
    examples = [
        "What documents do you have?",
        "Summarise the key points from all documents",
        "What does the report say about revenue?",
        "Compare Q2 and Q3 figures and calculate the growth rate",
        "What risks are mentioned across the documents?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending = ex

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show agent trace under each assistant message
        if msg["role"] == "assistant":
            trace_idx = i // 2
            if trace_idx < len(st.session_state.traces):
                trace = st.session_state.traces[trace_idx]
                tool_steps = [s for s in trace if s["action"] == "tool_call"]
                if tool_steps:
                    with st.expander(f"🔎 Agent trace — {len(tool_steps)} tool call(s)"):
                        for step in tool_steps:
                            st.markdown(f"**Step {step['step']}** · `{step['tool']}`"
                                        + (f" · ⏱ {step['duration_ms']} ms" if step["duration_ms"] else ""))
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption("Input")
                                st.json(step["input"], expanded=False)
                            with col2:
                                st.caption("Result preview")
                                st.text(step["result_preview"] or "—")
                            st.divider()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prefill = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask anything about your documents...") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = agent.run(user_input)
            trace_snapshot = [s.to_dict() for s in tracer.steps]

        st.markdown(answer)

        tool_steps = [s for s in trace_snapshot if s["action"] == "tool_call"]
        if tool_steps:
            with st.expander(f"🔎 Agent trace — {len(tool_steps)} tool call(s)"):
                for step in tool_steps:
                    st.markdown(f"**Step {step['step']}** · `{step['tool']}`"
                                + (f" · ⏱ {step['duration_ms']} ms" if step["duration_ms"] else ""))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(step["input"], expanded=False)
                    with col2:
                        st.caption("Result preview")
                        st.text(step["result_preview"] or "—")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.traces.append(trace_snapshot)

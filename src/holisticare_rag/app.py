"""Streamlit chat (extra ``ui``). Start it with ``holisticare ui``.

The index, the embedder and the chat model are made once per server process with
``st.cache_resource``. The chat history is in ``st.session_state`` and goes to the model as messages.
"""
from __future__ import annotations

import streamlit as st

from holisticare_rag.config import Settings
from holisticare_rag.safety import DISCLAIMER
from holisticare_rag.service import Assistant


@st.cache_resource
def _assistant() -> Assistant:
    return Assistant.from_settings(Settings.from_env())[0]


def main() -> None:
    st.set_page_config(page_title="holisticare-rag", layout="wide")
    st.title("holisticare-rag")
    st.warning(DISCLAIMER)
    assistant = _assistant()
    if "history" not in st.session_state:
        st.session_state.history = []
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.text(m["content"])
    q = st.chat_input("Ask a health question")
    if not q:
        return
    with st.chat_message("user"):
        st.text(q)
    a = assistant.ask(q, st.session_state.history)
    with st.chat_message("assistant"):
        st.text(a.text)  # plain text: no HTML or Markdown from the model is rendered
        for c in a.cautions:
            st.error(c)
        if a.citations:
            with st.expander("Sources and supporting passages"):
                for c in a.citations:
                    st.text(f"[{c.number}] {c.title} ({c.source}; {c.evidence})\n\"{c.quote}\"")
        for n in a.notes:
            st.caption(n)
    st.session_state.history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.text}]


if __name__ == "__main__":
    main()

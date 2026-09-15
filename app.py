"""Streamlit chat UI for xAI Grok."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.grok_client import MissingAPIKeyError, stream_chat

load_dotenv()

st.set_page_config(page_title="Grok Chatbot", page_icon="🤖", layout="centered")


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def new_chat() -> None:
    st.session_state.messages = []


def main() -> None:
    init_session()

    st.title("Grok Chatbot")
    st.caption("Streamlit + xAI Grok via the OpenAI-compatible API")

    with st.sidebar:
        st.header("Chat")
        if st.button("New chat", use_container_width=True):
            new_chat()
            st.rerun()
        st.markdown(
            "Set `XAI_API_KEY` in a local `.env` file "
            "(see `.env.example`). Optionally override `XAI_MODEL` / `XAI_BASE_URL`."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Message Grok…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_chat(st.session_state.messages))
        except MissingAPIKeyError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()
            return
        except Exception as exc:  # noqa: BLE001 — show API/network errors in UI
            st.error(f"Something went wrong talking to xAI: {exc}")
            st.session_state.messages.pop()
            return

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

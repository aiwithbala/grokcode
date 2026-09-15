"""Streamlit multimodal review UI for xAI Grok."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.documents import (
    ExtractedDocument,
    ExtractedImage,
    extract_document,
    extract_image,
    format_documents_for_prompt,
    is_image_upload,
)
from src.grok_client import (
    DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    MissingAPIKeyError,
    build_multimodal_user_message,
    stream_chat,
)
from src.youtube import (
    format_transcripts_for_prompt,
    gather_transcripts_from_text,
)

load_dotenv()

st.set_page_config(page_title="Grok Multimodal Review", page_icon="🤖", layout="centered")

UPLOAD_TYPES = [
    "pdf",
    "docx",
    "txt",
    "md",
    "csv",
    "json",
    "png",
    "jpg",
    "jpeg",
    "py",
    "js",
    "ts",
    "html",
    "css",
    "xml",
    "yaml",
    "yml",
    "toml",
    "log",
    "sql",
    "r",
    "go",
    "rs",
    "java",
    "c",
    "cpp",
    "h",
    "sh",
]


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents: list[ExtractedDocument] = []
    if "images" not in st.session_state:
        st.session_state.images: list[ExtractedImage] = []
    if "upload_errors" not in st.session_state:
        st.session_state.upload_errors: list[str] = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def new_chat() -> None:
    st.session_state.messages = []


def clear_uploads() -> None:
    st.session_state.documents = []
    st.session_state.images = []
    st.session_state.upload_errors = []
    st.session_state.uploader_key += 1


def process_uploads(uploaded_files) -> None:
    """Parse newly selected files into session attachments (replaces prior set)."""
    docs: list[ExtractedDocument] = []
    images: list[ExtractedImage] = []
    errors: list[str] = []

    for uf in uploaded_files or []:
        data = uf.getvalue()
        name = uf.name
        mime = getattr(uf, "type", None)

        if is_image_upload(name, mime):
            img = extract_image(name, data, mime_hint=mime)
            if img.error:
                errors.append(img.error)
            else:
                images.append(img)
        else:
            doc = extract_document(name, data)
            if doc.error:
                errors.append(doc.error)
            else:
                docs.append(doc)
                if doc.truncated:
                    errors.append(
                        f"'{doc.name}' was truncated to fit context limits "
                        "(beginning kept; see note in extracted text)."
                    )

    st.session_state.documents = docs
    st.session_state.images = images
    st.session_state.upload_errors = errors


def render_attachment_panel() -> None:
    docs: list[ExtractedDocument] = st.session_state.documents
    images: list[ExtractedImage] = st.session_state.images
    errors: list[str] = st.session_state.upload_errors

    st.subheader("Attachments in play")
    if not docs and not images and not errors:
        st.caption("No files attached yet. Upload documents or images above.")
        return

    if docs:
        st.markdown("**Documents**")
        for d in docs:
            flag = " (truncated)" if d.truncated else ""
            st.markdown(f"- `{d.name}`{flag}")
    if images:
        st.markdown("**Images** (sent to vision model)")
        for img in images:
            st.markdown(f"- `{img.name}` ({img.mime})")
    if errors:
        for err in errors:
            st.warning(err)

    if st.button("Clear uploads", use_container_width=True):
        clear_uploads()
        st.rerun()


def display_content(content: object) -> None:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                st.markdown(part.get("text", ""))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    st.caption("[attached image]")
                elif url:
                    st.image(url)
            else:
                st.markdown(str(part))
    else:
        st.markdown(str(content))


def main() -> None:
    init_session()

    st.title("Grok Multimodal Review")
    st.caption(
        "Upload docs/images, paste YouTube links, and give free-form instructions "
        "(summary, critique, elaborate script, …). Powered by xAI Grok."
    )

    with st.sidebar:
        st.header("Chat")
        if st.button("New chat", use_container_width=True):
            new_chat()
            st.rerun()

        st.divider()
        st.header("Uploads")
        uploaded = st.file_uploader(
            "Documents & images",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="PDF, DOCX, text/code, PNG/JPEG. Video files are out of scope.",
        )
        if uploaded is not None:
            # Re-process whenever the widget value changes (Streamlit reruns).
            process_uploads(uploaded)

        youtube_extra = st.text_input(
            "YouTube URL (optional)",
            placeholder="https://www.youtube.com/watch?v=…",
            help="Also detected automatically from your chat message.",
        )

        render_attachment_panel()

        st.divider()
        st.markdown(
            f"Set `XAI_API_KEY` in a local `.env` (see `.env.example`). "
            f"Default chat model: `{DEFAULT_MODEL}`. "
            f"When images are attached, vision default: `{DEFAULT_VISION_MODEL}` "
            f"(override both with `XAI_MODEL`)."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_content(message["content"])

    prompt = st.chat_input(
        "Instruction for Grok… (e.g. summarize, critique, elaborate a script)"
    )
    if not prompt:
        return

    # --- Gather YouTube context (message + optional sidebar field) ---
    yt_results = gather_transcripts_from_text(prompt)
    if youtube_extra and youtube_extra.strip():
        # Avoid duplicate fetches if the same URL is also in the prompt.
        already = {r.video_id for r in yt_results}
        extra = gather_transcripts_from_text(youtube_extra.strip())
        for r in extra:
            if r.video_id not in already:
                yt_results.append(r)

    yt_errors = [r.error for r in yt_results if r.error]
    yt_context = format_transcripts_for_prompt(yt_results)
    doc_context = format_documents_for_prompt(st.session_state.documents)
    image_urls = [img.data_url for img in st.session_state.images if img.data_url]
    has_images = bool(image_urls)

    user_api_message = build_multimodal_user_message(
        prompt,
        document_context=doc_context,
        youtube_context=yt_context,
        image_data_urls=image_urls,
    )

    # Display a friendly summary in the chat transcript.
    display_bits = [prompt]
    if st.session_state.documents:
        names = ", ".join(f"`{d.name}`" for d in st.session_state.documents)
        display_bits.append(f"_Documents: {names}_")
    if st.session_state.images:
        names = ", ".join(f"`{i.name}`" for i in st.session_state.images)
        display_bits.append(f"_Images: {names}_")
    ok_yt = [r for r in yt_results if r.transcript and not r.error]
    if ok_yt:
        display_bits.append(
            "_YouTube: " + ", ".join(f"`{r.video_id}`" for r in ok_yt) + "_"
        )
    display_text = "\n\n".join(display_bits)

    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)
        for err in yt_errors:
            st.warning(err)

    # History for the API: prior turns as plain text + current multimodal message.
    api_messages: list[dict] = []
    for msg in st.session_state.messages[:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append(user_api_message)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_chat(api_messages, has_images=has_images)
            )
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

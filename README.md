# grokcode — Streamlit + xAI Grok Multimodal Review

A Streamlit app that talks to **xAI Grok** through the OpenAI-compatible API (`https://api.x.ai/v1`). Review documents, images, and YouTube videos with free-form instructions (summarize, critique, elaborate a script, and more).

## Features

- Session message history and **New chat** in the sidebar
- Streaming assistant replies
- **Multi-file uploads**: PDF, DOCX, TXT/MD/CSV/JSON and common text/code files, plus PNG/JPEG images
- **YouTube**: paste a link in chat or the sidebar field; captions are fetched when available
- Free-form instructions alongside attachments
- Clear “attachments in play” panel with **Clear uploads**
- Friendly errors for missing API key, unreadable/unsupported files, and missing YouTube captions
- Uploaded **video files are out of scope** (use a YouTube link instead)

## Default models

| Mode | Default slug | Notes |
| --- | --- | --- |
| Chat / documents / YouTube | `grok-4.6` | xAI flagship chat model |
| With images attached | `grok-4.6` | Vision-capable; same flagship (image understanding via chat completions) |

Override either with `XAI_MODEL`. Images are sent as OpenAI-compatible `image_url` content parts (base64 data URLs).

## Setup

### 1. Get an xAI API key

1. Sign in at [https://console.x.ai/](https://console.x.ai/)
2. Create an API key
3. Keep it private — never commit it

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
XAI_API_KEY=your_key_here
```

Optional:

- `XAI_BASE_URL` — defaults to `https://api.x.ai/v1`
- `XAI_MODEL` — overrides both chat and vision defaults

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## How to use uploads / YouTube / instructions

1. **Upload** one or more PDFs, DOCX, text/code files, or PNG/JPEG images in the sidebar.
2. Confirm they appear under **Attachments in play**. Clear anytime with **Clear uploads**.
3. Optionally paste a **YouTube URL** in the sidebar field (or include it in your chat message).
4. Type a free-form instruction in the chat box, for example:
   - “Summarize the attached PDF in 5 bullets”
   - “Critique this slide screenshot for clarity”
   - “Turn the YouTube transcript into an elaborate script outline”
5. Send — document text and/or YouTube captions are included as context; images go to the vision-capable model.

Large documents and long transcripts are truncated with an explicit note so the request stays within context limits. Unknown file extensions are tried as UTF-8 text; binary/unknown types get a clear error. xAI image understanding accepts **PNG and JPEG** (max 20 MiB).

## Project layout

```
app.py                 # Thin Streamlit UI
src/grok_client.py     # OpenAI SDK client + multimodal message builder
src/documents.py       # PDF / DOCX / text / image extraction
src/youtube.py         # YouTube URL detection + transcript fetch
src/__init__.py
.env.example           # Env template (no secrets)
requirements.txt
```

## Notes

- No auth, database, or deploy config
- `.env` is gitignored; only `.env.example` is tracked
- Never commit real API keys

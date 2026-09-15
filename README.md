# grokcode — Streamlit + xAI Grok Chatbot

A minimal Streamlit chat app that talks to **xAI Grok** through the OpenAI-compatible API (`https://api.x.ai/v1`).

## Features

- Session message history in the Streamlit UI
- **New chat** button in the sidebar
- Streaming assistant replies
- Friendly error when `XAI_API_KEY` is missing

## Default model

**`grok-4.6`** — xAI’s current recommended chat / coding model on the API (as of 2026). Override with `XAI_MODEL` if you want another Grok slug.

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
- `XAI_MODEL` — defaults to `grok-4.6`

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

## Project layout

```
app.py                 # Streamlit chat UI
src/grok_client.py     # OpenAI SDK client for xAI
src/__init__.py
.env.example           # Env template (no secrets)
requirements.txt
```

## Notes

- No auth, database, or deploy config in this scaffold
- `.env` is gitignored; only `.env.example` is tracked

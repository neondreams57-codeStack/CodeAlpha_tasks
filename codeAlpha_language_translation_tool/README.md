# Lingua — Flask Translation App

A clean, deployable Flask app that translates text between 22 languages.
Supports **MyMemory** (free, no key), **Microsoft Azure Translator**, and **DeepL**.

---

## Quick Start (local)

```bash
git clone <your-repo>
cd lingua

# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — works immediately with no keys (uses MyMemory free tier)

# 4. Run
python app.py
# → http://localhost:5000
```

---

## API Backends

The app auto-selects a backend based on which keys are present in `.env`:

| Backend | Key variable | Free tier | Signup needed |
|---|---|---|---|
| **MyMemory** (default) | *(none)* | ~5 000 chars/day | No |
| MyMemory + email | `MYMEMORY_EMAIL` | ~50 000 chars/day | No |
| **Microsoft Azure** | `TRANSLATOR_API_KEY` | 2M chars/month | Yes |
| **DeepL** | `DEEPL_API_KEY` | 500 000 chars/month | Yes |

Priority: Azure > DeepL > MyMemory.

---


## Project Structure

```
lingua/
├── app.py                  # Flask app + translation logic
├── templates/
│   └── index.html          # Jinja2 template
├── static/
│   ├── css/style.css
│   └── js/main.js
├── requirements.txt
├── Procfile                # Heroku / Render / Railway
├── Dockerfile
└── .env.example
```

---

## REST API

### `POST /translate`

```json
// Request
{ "text": "Hello world", "src": "en", "tgt": "fr" }

// Response
{ "translation": "Bonjour le monde", "backend": "mymemory" }

// Error
{ "error": "Human-readable message" }
```

### `GET /health`

```json
{ "status": "ok", "backend": "mymemory" }
```

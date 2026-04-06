import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# MyMemory is used by default (free, no key required).
# To use Microsoft Azure Translator, set these in your .env:
#   TRANSLATOR_API_KEY=your_azure_key
#   TRANSLATOR_REGION=eastus   (or your region)
# To use DeepL, set:
#   DEEPL_API_KEY=your_deepl_key

AZURE_KEY    = os.getenv("TRANSLATOR_API_KEY", "")
AZURE_REGION = os.getenv("TRANSLATOR_REGION", "eastus")
DEEPL_KEY    = os.getenv("DEEPL_API_KEY", "")
MYMEMORY_EMAIL = os.getenv("MYMEMORY_EMAIL", "")   # optional, raises daily quota

LANGUAGES = [
    {"code": "en",    "name": "English"},
    {"code": "ar",    "name": "Arabic"},
    {"code": "zh",    "name": "Chinese (Simplified)"},
    {"code": "zh-TW", "name": "Chinese (Traditional)"},
    {"code": "fr",    "name": "French"},
    {"code": "de",    "name": "German"},
    {"code": "hi",    "name": "Hindi"},
    {"code": "id",    "name": "Indonesian"},
    {"code": "it",    "name": "Italian"},
    {"code": "ja",    "name": "Japanese"},
    {"code": "ko",    "name": "Korean"},
    {"code": "ms",    "name": "Malay"},
    {"code": "nl",    "name": "Dutch"},
    {"code": "pl",    "name": "Polish"},
    {"code": "pt",    "name": "Portuguese"},
    {"code": "ru",    "name": "Russian"},
    {"code": "es",    "name": "Spanish"},
    {"code": "sv",    "name": "Swedish"},
    {"code": "tr",    "name": "Turkish"},
    {"code": "uk",    "name": "Ukrainian"},
    {"code": "ur",    "name": "Urdu"},
    {"code": "vi",    "name": "Vietnamese"},
]


# ── Translation backends ───────────────────────────────────────────────────────

def translate_mymemory(text: str, src: str, tgt: str) -> str:
    """Free MyMemory API — no key needed, 5 000 chars/day anonymous."""
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": f"{src}|{tgt}"}
    if MYMEMORY_EMAIL:
        params["de"] = MYMEMORY_EMAIL

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    status = str(data.get("responseStatus", ""))
    if status == "200":
        return data["responseData"]["translatedText"]
    if status == "403":
        raise ValueError("MyMemory daily quota exceeded. Set MYMEMORY_EMAIL in .env to raise the limit.")
    raise ValueError(data.get("responseMessage", "MyMemory translation failed."))


def translate_azure(text: str, src: str, tgt: str) -> str:
    """Microsoft Azure Cognitive Services Translator."""
    url = "https://api.cognitive.microsofttranslator.com/translate"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_REGION,
        "Content-Type": "application/json",
    }
    params = {"api-version": "3.0", "from": src, "to": tgt}
    body = [{"text": text}]

    resp = requests.post(url, headers=headers, params=params, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()[0]["translations"][0]["text"]


def translate_deepl(text: str, src: str, tgt: str) -> str:
    """DeepL API (free or pro tier)."""
    base = "https://api-free.deepl.com" if DEEPL_KEY.endswith(":fx") else "https://api.deepl.com"
    url = f"{base}/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"}
    payload = {
        "text": [text],
        "source_lang": src.upper().split("-")[0],
        "target_lang": tgt.upper().replace("-", "_"),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["translations"][0]["text"]


def pick_backend() -> str:
    if AZURE_KEY:
        return "azure"
    if DEEPL_KEY:
        return "deepl"
    return "mymemory"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    backend = pick_backend()
    return render_template("index.html", languages=LANGUAGES, backend=backend)


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    src  = (data.get("src")  or "en").strip()
    tgt  = (data.get("tgt")  or "es").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text exceeds 5 000 character limit."}), 400
    if src == tgt:
        return jsonify({"error": "Source and target languages must differ."}), 400

    backend = pick_backend()
    try:
        if backend == "azure":
            result = translate_azure(text, src, tgt)
        elif backend == "deepl":
            result = translate_deepl(text, src, tgt)
        else:
            result = translate_mymemory(text, src, tgt)

        return jsonify({"translation": result, "backend": backend})

    except requests.HTTPError as e:
        return jsonify({"error": f"API error ({e.response.status_code}): {e.response.text[:200]}"}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except requests.RequestException as e:
        return jsonify({"error": f"Network error: {e}"}), 503


@app.route("/health")
def health():
    return jsonify({"status": "ok", "backend": pick_backend()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

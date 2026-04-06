# 🐍 Python FAQ Chatbot

> NLP-powered FAQ assistant using NLTK · TF-IDF · Cosine Similarity

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-27AE60?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-8E44AD?style=flat)

---

## Overview

A command-line FAQ chatbot for Python programming topics. Users type natural language questions and the bot finds the most relevant answer using **TF-IDF vectorization** and **cosine similarity** — no external APIs or large language models required.

---

## Features

- 📚 **20 curated Python FAQs** — installation, data types, OOP, memory management, decorators, and more
- 🔤 **NLP preprocessing pipeline** — tokenization, stopword removal, and lemmatization via NLTK
- 📊 **TF-IDF + bigrams** — richer semantic matching with unigram and bigram tokens
- 📐 **Cosine similarity scoring** — with a configurable confidence threshold
- 🎨 **Colour-coded CLI** — friendly output using colorama
- 🔧 **Easily extensible** — add new FAQs without changing any code

---

## Architecture

The chatbot processes each query through three layers:

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  1. NLP Preprocessor        │  lowercase → strip punctuation → tokenize
│     (NLTK)                  │  → remove stopwords → lemmatize
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. TF-IDF Vectorizer       │  transform query using the fitted
│     (scikit-learn)          │  FAQ vocabulary (ngram_range=(1,2))
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Cosine Similarity       │  compare query vector against all
│     Matcher                 │  FAQ vectors → return best match
└─────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/faq-chatbot.git
cd faq-chatbot

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install nltk scikit-learn numpy colorama
```

> NLTK data packages (`punkt`, `stopwords`, `wordnet`) are downloaded automatically on the first run.

---

## Usage

```bash
python faq_chatbot.py
```

### Available Commands

| Command | Description |
|---|---|
| `list` | Show all 20 FAQ topics in the knowledge base |
| `quit` / `exit` / `bye` | Exit the chatbot |
| `<any question>` | Find and display the best matching answer |

### Example Session

```
╔══════════════════════════════════════════════════════╗
║         🐍  Python FAQ Chatbot  🐍                   ║
║   Ask me anything about Python programming!          ║
║   Type  'list'  to see all topics                    ║
║   Type  'quit'  or  'exit'  to leave                 ║
╚══════════════════════════════════════════════════════╝

You: how do decorators work?

🔍 Matched: What are Python decorators?  (confidence: 78%)
Bot: Decorators are functions that wrap another function to extend or modify
     its behaviour without changing its source code. They use the @syntax sugar.
     Examples: @staticmethod, @classmethod, @property, or custom decorators
     for logging, caching, or access control.

You: list vs tuple difference

🔍 Matched: What is the difference between a list and a tuple?  (confidence: 65%)
Bot: Lists are mutable (you can change elements after creation) and defined
     with []. Tuples are immutable and defined with ()...
```

---

## How It Works

| Step | Stage | What happens |
|------|-------|-------------|
| 1 | Preprocess FAQs | All FAQ questions are cleaned and tokenized at startup |
| 2 | Fit TF-IDF | scikit-learn builds a weighted term matrix from FAQ tokens |
| 3 | Receive query | User types a question in the CLI |
| 4 | Preprocess query | Same NLTK pipeline applied to the user's input |
| 5 | Vectorize query | TF-IDF transforms the query using the fitted vocabulary |
| 6 | Compare | Cosine similarity computed against every FAQ vector |
| 7 | Return answer | Best match above threshold is displayed; fallback if below |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `nltk` | 3.8+ | Tokenization, stopword removal, lemmatization |
| `scikit-learn` | 1.0+ | TF-IDF vectorizer and cosine similarity |
| `numpy` | 1.21+ | Array operations for similarity scores |
| `colorama` | 0.4+ | Cross-platform colour output in the CLI |

---

## Project Structure

```
faq-chatbot/
├── faq_chatbot.py      # Main script — all logic in one file
├── README.md           # This file
└── requirements.txt    # Pinned dependency versions (optional)
```

---

## Extending the Knowledge Base

Open `faq_chatbot.py` and add new entries to the `FAQ_DATA` list. The vectorizer re-fits automatically on startup — no other changes needed.

```python
FAQ_DATA = [
    # ... existing FAQs ...
    {
        "question": "What is asyncio?",
        "answer": (
            "asyncio is Python's standard library for writing concurrent code "
            "using the async/await syntax. It is ideal for I/O-bound tasks such "
            "as web requests, file operations, or database queries."
        ),
    },
]
```

You can also tune the confidence threshold at the top of `FAQChatbot`:

```python
CONFIDENCE_THRESHOLD = 0.15  # raise to reduce false positives, lower to be more permissive
```

---

## License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it for any purpose with attribution.

---

*Built with Python, NLTK, and scikit-learn.*

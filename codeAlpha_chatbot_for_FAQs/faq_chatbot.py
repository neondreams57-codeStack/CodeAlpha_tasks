"""
FAQ Chatbot using NLP (NLTK + Cosine Similarity)
================================================
- FAQs about Python Programming (collected/curated)
- Preprocessing with NLTK: tokenization, stopword removal, lemmatization
- TF-IDF vectorization + Cosine Similarity for question matching
- Interactive CLI chatbot interface

Install dependencies:
    pip install nltk scikit-learn numpy colorama
"""

import re
import json
import nltk
import numpy as np
from colorama import Fore, Style, init
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Init ──────────────────────────────────────────────────────────────────────
init(autoreset=True)  # colorama

# Download required NLTK data (runs once)
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ── FAQ Dataset (Python Programming) ─────────────────────────────────────────
FAQ_DATA = [
    {
        "question": "What is Python?",
        "answer": (
            "Python is a high-level, interpreted, general-purpose programming language "
            "known for its clear syntax and readability. Created by Guido van Rossum and "
            "first released in 1991, Python supports multiple programming paradigms including "
            "procedural, object-oriented, and functional programming."
        ),
    },
    {
        "question": "How do I install Python?",
        "answer": (
            "Visit https://python.org/downloads and download the installer for your OS. "
            "On Windows, run the .exe and check 'Add Python to PATH'. "
            "On macOS/Linux you can also use a package manager: "
            "`brew install python3` (macOS) or `sudo apt install python3` (Ubuntu)."
        ),
    },
    {
        "question": "What is a virtual environment and why should I use one?",
        "answer": (
            "A virtual environment is an isolated Python environment that keeps project "
            "dependencies separate. Create one with `python -m venv venv`, activate it "
            "(`source venv/bin/activate` on Unix, `venv\\Scripts\\activate` on Windows), "
            "then install packages with pip. This prevents version conflicts between projects."
        ),
    },
    {
        "question": "What is pip?",
        "answer": (
            "pip is Python's standard package manager. It lets you install, upgrade, and "
            "remove third-party libraries from PyPI. Common commands: "
            "`pip install <package>`, `pip uninstall <package>`, `pip list`, "
            "`pip freeze > requirements.txt` to save dependencies."
        ),
    },
    {
        "question": "What is the difference between a list and a tuple?",
        "answer": (
            "Lists are mutable (you can change elements after creation) and defined with []. "
            "Tuples are immutable (cannot be changed after creation) and defined with (). "
            "Tuples are slightly faster and can be used as dictionary keys; lists are better "
            "when you need to add or remove items."
        ),
    },
    {
        "question": "What are Python decorators?",
        "answer": (
            "Decorators are functions that wrap another function to extend or modify its "
            "behaviour without changing its source code. They use the @syntax sugar. "
            "Example: @staticmethod, @classmethod, @property, or custom decorators for "
            "logging, caching, or access control."
        ),
    },
    {
        "question": "What is a lambda function in Python?",
        "answer": (
            "A lambda is an anonymous, inline function defined with the `lambda` keyword. "
            "Syntax: `lambda arguments: expression`. Example: `square = lambda x: x**2`. "
            "Lambdas are useful for short, throwaway functions, especially as arguments to "
            "map(), filter(), or sorted()."
        ),
    },
    {
        "question": "What is the difference between == and is in Python?",
        "answer": (
            "`==` checks value equality — whether two objects have the same value. "
            "`is` checks identity — whether two variables point to the exact same object "
            "in memory. For example, two lists with the same content are == but not `is`. "
            "Use `is` mainly for comparing with None: `if x is None:`."
        ),
    },
    {
        "question": "What are *args and **kwargs?",
        "answer": (
            "`*args` allows a function to accept any number of positional arguments as a tuple. "
            "`**kwargs` allows any number of keyword arguments as a dict. "
            "Example: `def func(*args, **kwargs)`. You can call it as "
            "`func(1, 2, name='Alice', age=30)`. They're useful for flexible APIs."
        ),
    },
    {
        "question": "What is list comprehension?",
        "answer": (
            "List comprehension is a concise way to create lists. "
            "Syntax: `[expression for item in iterable if condition]`. "
            "Example: `squares = [x**2 for x in range(10) if x % 2 == 0]`. "
            "It's generally faster and more readable than a for-loop with .append()."
        ),
    },
    {
        "question": "How does Python handle memory management?",
        "answer": (
            "Python uses automatic memory management via reference counting and a cyclic "
            "garbage collector. When an object's reference count drops to zero it is "
            "deallocated. The `gc` module handles reference cycles. You can inspect counts "
            "with `sys.getrefcount()`. CPython also uses memory pools (pymalloc) for "
            "small objects."
        ),
    },
    {
        "question": "What is the GIL?",
        "answer": (
            "The Global Interpreter Lock (GIL) is a mutex in CPython that allows only one "
            "thread to execute Python bytecode at a time. This simplifies memory management "
            "but limits multi-threaded CPU-bound performance. Use the `multiprocessing` "
            "module or async/await for parallelism. Note: Python 3.13+ introduces experimental "
            "no-GIL builds."
        ),
    },
    {
        "question": "What is the difference between append and extend in a list?",
        "answer": (
            "`list.append(x)` adds a single element x to the end. "
            "`list.extend(iterable)` adds every element of an iterable to the end. "
            "Example: `[1,2].append([3,4])` → `[1,2,[3,4]]` (nested list), "
            "but `[1,2].extend([3,4])` → `[1,2,3,4]`."
        ),
    },
    {
        "question": "How do I read and write files in Python?",
        "answer": (
            "Use the built-in `open()` function. Always prefer the `with` statement to "
            "ensure the file is closed automatically. "
            "Read: `with open('file.txt', 'r') as f: content = f.read()`. "
            "Write: `with open('file.txt', 'w') as f: f.write('hello')`. "
            "Modes: 'r' read, 'w' write, 'a' append, 'rb'/'wb' for binary."
        ),
    },
    {
        "question": "What are Python generators?",
        "answer": (
            "Generators are functions that yield values one at a time using the `yield` "
            "keyword, producing a lazy iterator. They save memory for large sequences. "
            "Example: `def count_up(n): yield from range(n)`. "
            "Generator expressions work like comprehensions: `(x**2 for x in range(100))`."
        ),
    },
    {
        "question": "What is exception handling in Python?",
        "answer": (
            "Use try/except/else/finally blocks. `try` wraps risky code, `except` catches "
            "specific exceptions, `else` runs if no exception occurred, `finally` always runs. "
            "Raise exceptions with `raise`. Define custom exceptions by subclassing Exception. "
            "Best practice: catch specific exceptions, not bare `except:`."
        ),
    },
    {
        "question": "What is PEP 8?",
        "answer": (
            "PEP 8 is Python's official style guide covering code formatting, naming "
            "conventions, and best practices. Key rules: 4-space indentation, max 79 "
            "characters per line, snake_case for variables/functions, PascalCase for classes, "
            "UPPER_CASE for constants. Use tools like `flake8` or `black` to enforce it."
        ),
    },
    {
        "question": "What is the difference between Python 2 and Python 3?",
        "answer": (
            "Python 2 reached end-of-life in January 2020. Key Python 3 differences: "
            "`print` is a function not a statement, division of integers returns float by default, "
            "strings are Unicode by default, `range()` returns an iterator, "
            "many libraries are Python 3 only. Always use Python 3 for new projects."
        ),
    },
    {
        "question": "How do I use classes and OOP in Python?",
        "answer": (
            "Define a class with `class ClassName:`. Use `__init__` for initialisation, "
            "`self` to refer to the instance. Support inheritance: `class Dog(Animal):`. "
            "Key OOP concepts: encapsulation, inheritance, polymorphism. "
            "Use `@property` for getters/setters, `__str__` / `__repr__` for string representation."
        ),
    },
    {
        "question": "What are Python modules and packages?",
        "answer": (
            "A module is a single .py file you can import. A package is a directory with an "
            "`__init__.py` file containing multiple modules. Import with `import module` or "
            "`from package.module import function`. Use `__name__ == '__main__'` guard to "
            "prevent code from running on import. The standard library has hundreds of built-in modules."
        ),
    },
]


# ── NLP Preprocessor ─────────────────────────────────────────────────────────
class TextPreprocessor:
    """Tokenize, clean, remove stopwords, and lemmatize text."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)          # remove punctuation
        tokens = word_tokenize(text)                       # tokenize
        tokens = [t for t in tokens if t not in self.stop_words]  # remove stopwords
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]   # lemmatize
        return " ".join(tokens)


# ── FAQ Chatbot ───────────────────────────────────────────────────────────────
class FAQChatbot:
    """
    Matches user questions to FAQs using TF-IDF + Cosine Similarity.
    """

    CONFIDENCE_THRESHOLD = 0.15  # minimum similarity to return an answer

    def __init__(self, faq_data: list[dict]):
        self.faq_data = faq_data
        self.preprocessor = TextPreprocessor()

        # Preprocess all FAQ questions
        self.processed_questions = [
            self.preprocessor.preprocess(item["question"]) for item in faq_data
        ]

        # Fit TF-IDF on FAQ questions
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def get_response(self, user_input: str) -> tuple[str, float, str]:
        """
        Returns (answer, confidence_score, matched_question).
        """
        processed_input = self.preprocessor.preprocess(user_input)
        if not processed_input.strip():
            return "Please type a question!", 0.0, ""

        # Vectorize the user query and compute cosine similarity
        query_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self.CONFIDENCE_THRESHOLD:
            return (
                "Sorry, I couldn't find a matching answer. "
                "Try rephrasing or ask about Python installation, lists, decorators, etc.",
                best_score,
                "",
            )

        matched_q = self.faq_data[best_idx]["question"]
        answer = self.faq_data[best_idx]["answer"]
        return answer, best_score, matched_q

    def get_all_questions(self) -> list[str]:
        return [item["question"] for item in self.faq_data]


# ── CLI Interface ─────────────────────────────────────────────────────────────
def print_banner():
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════╗
║         🐍  Python FAQ Chatbot  🐍                   ║
║   Ask me anything about Python programming!          ║
║   Type  'list'  to see all topics                    ║
║   Type  'quit'  or  'exit'  to leave                 ║
╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def run_cli(bot: FAQChatbot):
    print_banner()

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.YELLOW}Goodbye! Happy coding 🐍{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "bye"):
            print(f"{Fore.YELLOW}Goodbye! Happy coding 🐍{Style.RESET_ALL}")
            break

        if cmd == "list":
            print(f"\n{Fore.CYAN}📋 Available FAQ topics:{Style.RESET_ALL}")
            for i, q in enumerate(bot.get_all_questions(), 1):
                print(f"  {Fore.WHITE}{i:2}. {q}{Style.RESET_ALL}")
            print()
            continue

        answer, score, matched_q = bot.get_response(user_input)

        if matched_q:
            print(f"\n{Fore.MAGENTA}🔍 Matched: {matched_q}  (confidence: {score:.0%}){Style.RESET_ALL}")

        print(f"{Fore.BLUE}Bot: {Style.RESET_ALL}{answer}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising FAQ Chatbot …")
    bot = FAQChatbot(FAQ_DATA)
    run_cli(bot)

import sys

# Streamlit Cloud / Linux ships an older system sqlite3; swap in pysqlite3 if installed.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get_config(key: str, default: str = "") -> str:
    """Reads config from environment variables (.env) first, then Streamlit st.secrets."""
    val = os.getenv(key)
    if val is not None and str(val).strip():
        return str(val).strip()
    try:
        import streamlit as st

        if key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val is not None and str(secret_val).strip():
                return str(secret_val).strip()
    except Exception:
        pass
    return default


_raw_chroma_dir = _get_config("CHROMA_DIR", "")
CHROMA_DIR = (
    str(Path(_raw_chroma_dir))
    if _raw_chroma_dir and Path(_raw_chroma_dir).is_absolute()
    else str(BASE_DIR / (_raw_chroma_dir or "chroma_durgeshnandini"))
)
CACHE_FILE = BASE_DIR / _get_config("CACHE_FILE", "durgeshnandini_chapters.json")
EMBEDDING_MODEL = _get_config("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# Strictly allow ONLY these 4 models (ordered by highest discount first: 60% -> 56%)
ALLOWED_MODELS = [
    ("deepseek-v4.1-flash-1017", 70.0),  # 70.00% off
    ("gpt-5.6-luna", 60.0),              # 60.00% off
    ("gpt-5.6-terra", 60.0),             # 60.00% off
    ("gpt-oss-120b", 60.0),              # 60.00% off
]

MODEL_COOLDOWN_SECONDS = 900

_VECTORSTORE = None
_VECTORSTORE_LOCK = threading.Lock()
_CACHED_CHAPTERS = None
_RANKED_MODELS = None
_EXHAUSTED_MODELS: dict[str, float] = {}


def get_api_key() -> str:
    return _get_config("API_KEY", "")


def get_base_url() -> str:
    return _get_config("BASE_URL", "https://api.cheaperinference.com/v1")


def get_ranked_models(force_refresh: bool = False) -> list[str]:
    """
    Returns ONLY the models defined in ALLOWED_MODELS, sorted by highest discount first.
    Optionally checks /models to filter out any temporarily unlisted model IDs.
    """
    global _RANKED_MODELS
    if _RANKED_MODELS is not None and not force_refresh:
        return _RANKED_MODELS

    sorted_allowed = sorted(
        enumerate(ALLOWED_MODELS),
        key=lambda item: (-item[1][1], item[0]),
    )
    ordered_ids = [mid for _, (mid, _) in sorted_allowed]
    allowed_lookup = {mid.lower(): mid for mid in ordered_ids}

    api_key = get_api_key()
    base_url = get_base_url()

    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        raw_list = payload.get("data", payload) if isinstance(payload, dict) else payload

        if isinstance(raw_list, list):
            live_allowed = set()
            for item in raw_list:
                if isinstance(item, dict) and item.get("id"):
                    mid = str(item["id"]).strip().lower()
                    if mid in allowed_lookup:
                        live_allowed.add(allowed_lookup[mid])

            if live_allowed:
                _RANKED_MODELS = [mid for mid in ordered_ids if mid in live_allowed]
    except Exception as e:
        print(f"Could not fetch live /models list ({e}); using ALLOWED_MODELS directly.", flush=True)

    if not _RANKED_MODELS:
        _RANKED_MODELS = ordered_ids

    print(f"Active model priority: {_RANKED_MODELS}", flush=True)
    return _RANKED_MODELS


def get_available_models() -> list[str]:
    """Returns allowed models skipping any currently in cooldown after errors/exhaustion."""
    ranked = get_ranked_models()
    now = time.time()

    expired = [m for m, ts in _EXHAUSTED_MODELS.items() if now - ts >= MODEL_COOLDOWN_SECONDS]
    for m in expired:
        del _EXHAUSTED_MODELS[m]

    available = [m for m in ranked if m not in _EXHAUSTED_MODELS]
    if not available:
        _EXHAUSTED_MODELS.clear()
        available = list(ranked)
    return available


def mark_model_exhausted(model_name: str, reason: str):
    _EXHAUSTED_MODELS[model_name] = time.time()
    print(f"Switching away from '{model_name}' ({reason}). Choosing next model...", flush=True)


def get_cached_chapters() -> list[dict]:
    global _CACHED_CHAPTERS
    if _CACHED_CHAPTERS is None and CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _CACHED_CHAPTERS = json.load(f)
    return _CACHED_CHAPTERS or []


def _build_documents_from_chapters(chapters: list[dict]) -> list[Document]:
    """Creates passage-prefixed chunks from cached chapters if ChromaDB needs initialization."""
    docs: list[Document] = []
    for idx, ch in enumerate(chapters, start=1):
        paragraphs = [p.strip() for p in ch.get("text", "").split("\n\n") if p.strip()]
        chapter_title = ch.get("chapter", f"পরিচ্ছেদ {idx}")
        source_url = ch.get("source_url", "")

        if not paragraphs:
            continue

        step = 2
        window_size = 3
        for i in range(0, len(paragraphs), step):
            chunk_text = "\n\n".join(paragraphs[i : i + window_size])
            docs.append(
                Document(
                    page_content=f"passage: [{chapter_title}]\n{chunk_text}",
                    metadata={
                        "chapter": chapter_title,
                        "chapter_num": idx,
                        "source_url": source_url,
                    },
                )
            )
            if i + window_size >= len(paragraphs):
                break
    return docs


def get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    with _VECTORSTORE_LOCK:
        if _VECTORSTORE is not None:
            return _VECTORSTORE

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="durgeshnandini",
        )

        try:
            existing_count = vs._collection.count()
        except Exception:
            existing_count = 0

        if existing_count == 0:
            chapters = get_cached_chapters()
            if chapters:
                print("ChromaDB empty/missing; building index from durgeshnandini_chapters.json...", flush=True)
                docs = _build_documents_from_chapters(chapters)
                if docs:
                    vs.add_documents(docs)
            else:
                raise FileNotFoundError(
                    f"Neither '{CHROMA_DIR}' nor '{CACHE_FILE}' was found. "
                    "Ensure at least one is committed to your repository."
                )

        _VECTORSTORE = vs
    return _VECTORSTORE


STOPWORDS = {
    "কাদের", "কাহাদের", "কাহার", "কার", "কে", "কি", "কী", "কোথায়", "কখন",
    "কেন", "কিভাবে", "কত", "কোন", "সাথে", "সঙ্গে", "হয়েছিল", "হইয়াছিল",
    "ছিল", "ছিলেন", "আছে", "নিয়েছিলেন", "দিয়েছিলেন", "বলেছিলেন", "কহিলেন",
    "কহিল", "হইলেন", "করিলেন", "এই", "সেই", "তাঁহার", "তাহার", "তাহা",
    "এবং", "ও", "বা", "থেকে", "হতে", "হইতে", "জন্য", "উপন্যাসে", "দুর্গেশনন্দিনী",
    "প্রথম", "দ্বিতীয়", "খণ্ড", "পরিচ্ছেদ", "কিন্তু", "তবে", "যদি", "আমি", "তুমি",
    "আপনি", "তিনি", "আমরা", "তাহারা", "এক", "একটি", "কোনো", "সকল",
}


def extract_bengali_keywords(text: str, top_k: int = 10) -> list[str]:
    """Extracts the most frequent meaningful Bengali word stems."""
    words = re.findall(r"[\u0980-\u09FF]+", text)
    counts = Counter()
    for w in words:
        stem = re.sub(r"(ের|দের|কে|তে|ে|র)$", "", w)
        if len(stem) >= 3 and w not in STOPWORDS and stem not in STOPWORDS:
            counts[stem] += 1
    return [kw for kw, _ in counts.most_common(top_k)]


def diverse_keyword_search(keywords: list[str], max_per_chapter: int = 1, top_n: int = 6) -> list[Document]:
    """
    Searches cached chapters with a strict per-chapter cap so one chapter
    cannot drown out earlier chapters.
    """
    chapters = get_cached_chapters()
    if not chapters or not keywords:
        return []

    scored = []
    for idx, ch in enumerate(chapters, start=1):
        paragraphs = ch["text"].split("\n\n")
        best_for_chapter = []
        for i in range(len(paragraphs)):
            window = "\n\n".join(paragraphs[i : i + 3])
            distinct = sum(1 for kw in keywords if kw in window)
            total = sum(window.count(kw) for kw in keywords)
            if distinct >= 2 or total >= 2:
                score = distinct * 20 + total
                best_for_chapter.append((
                    score,
                    Document(
                        page_content=f"passage: [{ch['chapter']}]\n{window}",
                        metadata={
                            "chapter": ch["chapter"],
                            "chapter_num": idx,
                            "source_url": ch["source_url"],
                        },
                    ),
                ))
        best_for_chapter.sort(key=lambda x: x[0], reverse=True)
        scored.extend(best_for_chapter[:max_per_chapter])

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def retrieve_relevant_docs(question: str, k: int = 12) -> list[Document]:
    """
    Multi-chapter hybrid retrieval:
    1. Diverse keyword search across different chapters (max 1 window per chapter).
    2. Dense vector search with per-chapter diversity cap (max 2 chunks per chapter).
    3. Hop-2 entity expansion using the most frequent character/place stems found in Hop 1.
    """
    vectorstore = get_vectorstore()
    q_keywords = extract_bengali_keywords(question, top_k=8)

    kw_docs = diverse_keyword_search(q_keywords, max_per_chapter=1, top_n=5)
    raw_dense = vectorstore.similarity_search(f"query: {question}", k=k)

    hop1_text = "\n".join(d.page_content for d in kw_docs + raw_dense[:4])
    hop2_keywords = [
        kw for kw in extract_bengali_keywords(hop1_text, top_k=12)
        if kw not in q_keywords
    ][:5]
    hop2_docs = vectorstore.similarity_search(
        f"query: {question} {' '.join(hop2_keywords)}", k=4
    )

    merged = []
    seen_fp = set()
    chapter_counts = Counter()

    for doc in kw_docs + raw_dense + hop2_docs:
        ch = doc.metadata.get("chapter", "")
        fp = doc.page_content[:120]
        if fp not in seen_fp and chapter_counts[ch] < 2:
            seen_fp.add(fp)
            chapter_counts[ch] += 1
            merged.append(doc)
        if len(merged) >= 9:
            break

    merged.sort(key=lambda d: d.metadata.get("chapter_num", 999))
    return merged


SYSTEM_PROMPT = """আপনি বঙ্কিমচন্দ্র চট্টোপাধ্যায়ের ঐতিহাসিক উপন্যাস 'দুর্গেশনন্দিনী'-এর একজন নির্ভুল সাহিত্য সহায়ক।
নিচে উপন্যাসের একাধিক পরিচ্ছেদ থেকে কালানুক্রমিকভাবে প্রাসঙ্গিক অংশ (Context) দেওয়া হয়েছে।

কঠোর নিয়মাবলী:
১. শুধুমাত্র প্রদত্ত প্রসঙ্গের (Context) তথ্যের ওপর ভিত্তি করে উত্তর দিন।
২. সকল পরিচ্ছেদের তথ্য একত্র করে সম্পূর্ণ উত্তর দিন। কোনো ঘটনা বা সাক্ষাতে যদি একাধিক ব্যক্তি জড়িত থাকে (যেমন—প্রথম পরিচ্ছেদে দুই রমণী এবং পরবর্তী পরিচ্ছেদে তাদের পরিচয় তিলোত্তমা ও বিমলা), তবে **প্রত্যেক ব্যক্তির নাম ও বইয়ে উল্লিখিত সঠিক পরিচয়** স্পষ্টভাবে উল্লেখ করুন। কেবল একজনকে লিখে অন্যজনকে বাদ দেবেন না।
৩. বইয়ের প্রসঙ্গে স্পষ্টভাবে উল্লেখ না থাকলে কোনো সম্পর্ক (যেমন—কাউকে কারো স্ত্রী বানানো) নিজের থেকে অনুমান করবেন না।
৪. প্রতিটি উত্তরের শেষে ব্যবহৃত খণ্ড ও পরিচ্ছেদের নাম উল্লেখ করুন। যেমন: [সূত্র: প্রথম খণ্ড - প্রথম পরিচ্ছেদ, দশম পরিচ্ছেদ, ষোড়শ পরিচ্ছেদ]।
৫. যদি প্রদত্ত প্রসঙ্গে প্রশ্নের উত্তর একেবারেই না থাকে, তবে বলুন: "দুঃখিত, 'দুর্গেশনন্দিনী' উপন্যাসের প্রদত্ত অংশে এই প্রশ্নের কোনো উত্তর পাওয়া যায়নি।"

প্রসঙ্গ (Context):
{context}
"""


def format_docs(docs: list[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        clean_text = doc.page_content.replace("passage: ", "", 1)
        chapter = doc.metadata.get("chapter", "অজানা পরিচ্ছেদ")
        formatted_chunks.append(f"=== {chapter} ===\n{clean_text}")
    return "\n\n".join(formatted_chunks)


def ask_book(question: str) -> dict:
    api_key = get_api_key()
    base_url = get_base_url()
    if not api_key:
        raise ValueError(
            "API_KEY is not set. Add API_KEY=... to .env (local) or App Settings -> Secrets (Streamlit Cloud)."
        )

    retrieved_docs = retrieve_relevant_docs(question, k=12)
    context_text = format_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    messages = prompt.format_messages(context=context_text, question=question)

    candidate_models = get_available_models()
    errors = []

    for model_name in candidate_models:
        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                max_tokens=4096,
                max_retries=1,
                timeout=60,
            )
            response = llm.invoke(messages)
            answer = (response.content or "").strip()

            if not answer:
                finish_reason = response.response_metadata.get("finish_reason", "empty")
                reason = f"empty response / finish_reason={finish_reason}"
                errors.append(f"{model_name}: {reason}")
                mark_model_exhausted(model_name, reason)
                continue

            return {
                "model": model_name,
                "answer": answer,
                "sources": retrieved_docs,
            }

        except AuthenticationError:
            raise
        except RateLimitError as e:
            reason = "429 rate limit / token quota finished"
            errors.append(f"{model_name}: {reason} ({e})")
            mark_model_exhausted(model_name, reason)
        except APIStatusError as e:
            reason = f"HTTP {e.status_code}"
            errors.append(f"{model_name}: {reason} ({e})")
            mark_model_exhausted(model_name, reason)
        except (BadRequestError, APITimeoutError, APIConnectionError) as e:
            reason = type(e).__name__
            errors.append(f"{model_name}: {reason} ({e})")
            mark_model_exhausted(model_name, reason)

    raise RuntimeError(
        "All candidate models failed or ran out of tokens:\n" + "\n".join(errors)
    )
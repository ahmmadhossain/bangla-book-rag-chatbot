import json
import os
import re
import time
from collections import Counter

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

load_dotenv()

CHROMA_DIR = "./chroma_durgeshnandini"
CACHE_FILE = "./durgeshnandini_chapters.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://api.cheaperinference.com/v1").strip()
# Set ONLY_FREE=true in .env if you want to rotate strictly among Free models
ONLY_FREE = os.getenv("ONLY_FREE", "false").strip().lower() == "true"

# Highest-discount text models from CheaperInference (Free first, then highest % off)
# Note: parallel-decisions-fast is excluded because it is a Decision-only model, not Text/Chat.
HIGHEST_DISCOUNT_FALLBACK_MODELS = [
    ("beam-free", 30.0),                    # Free
    ("nemotron-3.5-lightning-free", 30.0),  # Free
    ("deepseek-v4.1-flash-1017", 56.0),      # 56.00% off
    ("longcat-2.5", 64.0),                   # 64.00% off
    ("gpt-5.6-luna", 60.0),                  # 60.00% off
    ("gpt-5.6-terra", 60.0),                 # 60.00% off
    ("gpt-oss-120b", 60.0),                 # 60.00% off
    ("flint-v1.0", 30.0),                   # Free
    ("glyph-cluster", 30.0),                # Free
    ("ampersand-v0.1", 30.0)                # Free
]

NON_CHAT_KEYWORDS = (
    "embed", "rerank", "whisper", "tts", "dall-e", "flux", "sdxl",
    "moderation", "audio", "video", "image", "transcrib", "decision", "retrieval",
)

# How long to skip an exhausted/rate-limited model before retrying it (seconds)
MODEL_COOLDOWN_SECONDS = 900

_VECTORSTORE = None
_CACHED_CHAPTERS = None
_RANKED_MODELS = None
_EXHAUSTED_MODELS: dict[str, float] = {}


def _is_text_chat_model(model_id: str, meta: dict) -> bool:
    lower_id = model_id.lower()
    if any(kw in lower_id for kw in NON_CHAT_KEYWORDS):
        return False

    # Check optional capability/type metadata returned by the provider
    model_type = str(meta.get("type", "") or meta.get("category", "")).lower()
    if model_type and model_type not in ("text", "chat", "llm", "completion", "model"):
        return False

    caps = meta.get("capabilities") or meta.get("modalities")
    if isinstance(caps, list) and caps:
        cap_strs = {str(c).lower() for c in caps}
        if not cap_strs.intersection({"text", "chat", "completion"}):
            return False

    return True


def _extract_discount_score(model_id: str, meta: dict, fallback_scores: dict[str, float]) -> float:
    """Returns discount percentage (100.0 for Free, 0.0-99.9 for discounted paid models)."""
    lower_id = model_id.lower()
    if meta.get("free") is True or meta.get("is_free") is True or "free" in lower_id:
        return 100.0

    # Check pricing fields if the API returns them
    pricing = meta.get("pricing") if isinstance(meta.get("pricing"), dict) else {}
    in_price = pricing.get("prompt", pricing.get("input", meta.get("input_price")))
    out_price = pricing.get("completion", pricing.get("output", meta.get("output_price")))
    if in_price is not None and out_price is not None:
        try:
            if float(in_price) == 0.0 and float(out_price) == 0.0:
                return 100.0
        except (TypeError, ValueError):
            pass

    # Check explicit discount fields if returned by the API
    for key in ("discount", "discount_percent", "discount_pct", "discount_percentage"):
        val = meta.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            if "free" in val.lower():
                return 100.0
            m = re.search(r"(\d+(?:\.\d+)?)", val)
            if m:
                num = float(m.group(1))
                return num * 100.0 if 0 < num <= 1.0 else num
        elif isinstance(val, (int, float)):
            return float(val) * 100.0 if 0 < float(val) <= 1.0 else float(val)

    return fallback_scores.get(model_id, -1.0)


def get_ranked_models(force_refresh: bool = False) -> list[str]:
    """
    Auto-loads text models from BASE_URL/models and sorts them by highest discount
    (Free first, then highest % discount). Falls back to HIGHEST_DISCOUNT_FALLBACK_MODELS
    if /models does not expose discount metadata or is unreachable.
    """
    global _RANKED_MODELS
    if _RANKED_MODELS is not None and not force_refresh:
        return _RANKED_MODELS

    fallback_scores = dict(HIGHEST_DISCOUNT_FALLBACK_MODELS)
    fallback_order = {mid: i for i, (mid, _) in enumerate(HIGHEST_DISCOUNT_FALLBACK_MODELS)}

    live_models: list[tuple[str, dict]] = []
    try:
        resp = requests.get(
            f"{BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        raw_list = payload.get("data", payload) if isinstance(payload, dict) else payload
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict) and item.get("id"):
                    mid = str(item["id"]).strip()
                    if _is_text_chat_model(mid, item):
                        live_models.append((mid, item))
    except Exception as e:
        print(f"Could not fetch live /models list ({e}); using built-in highest-discount list.", flush=True)

    if live_models:
        scored = []
        for mid, meta in live_models:
            discount = _extract_discount_score(mid, meta, fallback_scores)
            if ONLY_FREE and discount < 100.0:
                continue
            # Sort by: highest discount DESC, known dashboard order ASC, model name ASC
            scored.append((discount, fallback_order.get(mid, 9999), mid))

        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        _RANKED_MODELS = [mid for _, _, mid in scored]

    if not _RANKED_MODELS:
        _RANKED_MODELS = [
            mid for mid, disc in HIGHEST_DISCOUNT_FALLBACK_MODELS
            if not ONLY_FREE or disc >= 100.0
        ]

    print(f"Auto-selected model priority (highest discount first): {_RANKED_MODELS[:6]}...", flush=True)
    return _RANKED_MODELS


def get_available_models() -> list[str]:
    """Returns ranked models skipping any currently in cooldown after token exhaustion."""
    ranked = get_ranked_models()
    now = time.time()

    # Clear expired cooldowns
    expired = [m for m, ts in _EXHAUSTED_MODELS.items() if now - ts >= MODEL_COOLDOWN_SECONDS]
    for m in expired:
        del _EXHAUSTED_MODELS[m]

    available = [m for m in ranked if m not in _EXHAUSTED_MODELS]
    if not available:
        # All models were exhausted; reset cooldowns and try from the top again
        _EXHAUSTED_MODELS.clear()
        available = list(ranked)
    return available


def mark_model_exhausted(model_name: str, reason: str):
    _EXHAUSTED_MODELS[model_name] = time.time()
    print(f"Switching away from '{model_name}' ({reason}). Choosing next model...", flush=True)


def get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _VECTORSTORE = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="durgeshnandini",
        )
    return _VECTORSTORE


def get_cached_chapters() -> list[dict]:
    global _CACHED_CHAPTERS
    if _CACHED_CHAPTERS is None and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _CACHED_CHAPTERS = json.load(f)
    return _CACHED_CHAPTERS or []


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
    (e.g. ষোড়শ পরিচ্ছেদ: শৈলেশ্বর-সাক্ষাৎ) cannot drown out earlier chapters.
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

    # Hop 1: Diverse multi-chapter keyword + dense retrieval
    kw_docs = diverse_keyword_search(q_keywords, max_per_chapter=1, top_n=5)
    raw_dense = vectorstore.similarity_search(f"query: {question}", k=k)

    # Hop 2: Find frequent proper nouns/terms in Hop 1 and pull their cross-chapter context
    hop1_text = "\n".join(d.page_content for d in kw_docs + raw_dense[:4])
    hop2_keywords = [
        kw for kw in extract_bengali_keywords(hop1_text, top_k=12)
        if kw not in q_keywords
    ][:5]
    hop2_docs = vectorstore.similarity_search(
        f"query: {question} {' '.join(hop2_keywords)}", k=4
    )

    # Merge with strict per-chapter cap (max 2 chunks per chapter) and deduplication
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

    # Present excerpts in chronological story order so the LLM follows plot reveals cleanly
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
    if not API_KEY:
        raise ValueError("API_KEY is not set. Add API_KEY=... to your .env file or environment.")

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
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=0.0,
                max_tokens=1536,
                max_retries=1,
                timeout=45,
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
            # Invalid API key applies to all models; fail immediately
            raise
        except RateLimitError as e:
            reason = "429 rate limit / token quota finished"
            errors.append(f"{model_name}: {reason} ({e})")
            mark_model_exhausted(model_name, reason)
        except APIStatusError as e:
            # Covers 402 (credits exhausted), 403, 404, 500/502/503/529
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
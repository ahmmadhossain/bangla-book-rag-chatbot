import json
import os
import re
import time
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup

API_URL = "https://bn.wikisource.org/w/api.php"
BASE_WIKI_URL = "https://bn.wikisource.org/wiki/"
CACHE_FILE = "./durgeshnandini_chapters.json"

HEADERS = {
    "User-Agent": "DurgeshnandiniRAGBot/1.0"
}

BENGALI_ORDINALS = {
    "প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুর্থ": 4, "পঞ্চম": 5,
    "ষষ্ঠ": 6, "সপ্তম": 7, "অষ্টম": 8, "নবম": 9, "দশম": 10,
    "একাদশ": 11, "দ্বাদশ": 12, "ত্রয়োদশ": 13, "চতুর্দশ": 14, "চতুর্দ্দশ": 14,
    "পঞ্চদশ": 15, "ষোড়শ": 16, "সপ্তদশ": 17, "অষ্টাদশ": 18, "ঊনবিংশ": 19,
    "বিংশ": 20, "একবিংশ": 21, "দ্বাবিংশ": 22, "ত্রয়োবিংশ": 23, "চতুর্বিংশ": 24,
    "পঞ্চবিংশ": 25,
}


def chapter_sort_key(chapter_label: str) -> tuple[int, int]:
    """Sorts chapters into true reading order: (volume_num, chapter_num)."""
    vol = 1 if "প্রথম খণ্ড" in chapter_label else 2 if "দ্বিতীয় খণ্ড" in chapter_label else 3
    ch_num = 999
    m = re.search(r"([\u0980-\u09FF]+)\s+পরিচ্ছেদ", chapter_label)
    if m:
        ch_num = BENGALI_ORDINALS.get(m.group(1), 999)
    return (vol, ch_num)


def api_get_with_retry(params: dict, max_retries: int = 6) -> dict:
    for attempt in range(max_retries):
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10 * (attempt + 1)))
            print(f"Rate limited (429). Sleeping {retry_after}s...", flush=True)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exceeded max retries due to Wikisource rate limiting.")


def parse_chapter_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for unwanted in soup.find_all(
        ["table", "div", "span"],
        class_=re.compile(r"(ws-noexport|headertemplate|footertemplate|pagenum|mw-editsection|noprint)")
    ):
        unwanted.decompose()

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 15:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def fetch_all_chapters() -> list[dict]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached:
            cached.sort(key=lambda item: chapter_sort_key(item["chapter"]))
            # Persist chronological order so rag_chain.py reads chapters in the same order
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)
            print(f"Loaded {len(cached)} cached chapters in chronological order.", flush=True)
            return cached

    print("Discovering all chapter subpages of দুর্গেশনন্দিনী via Wikisource API...", flush=True)
    titles = []
    seen = set()

    for ns in [0, 102]:
        data = api_get_with_retry({
            "action": "query",
            "list": "allpages",
            "apprefix": "দুর্গেশনন্দিনী/",
            "apnamespace": ns,
            "aplimit": "max",
            "format": "json",
        })
        for page in data.get("query", {}).get("allpages", []):
            title = page["title"]
            if "পরিচ্ছেদ" in title and title not in seen:
                seen.add(title)
                titles.append(title)
        time.sleep(1.0)

    print(f"Found {len(titles)} chapters. Downloading rendered chapter text...", flush=True)

    chapters = []
    for idx, title in enumerate(titles, start=1):
        clean_label = (
            title.replace("সাহিত্য:", "")
            .replace("দুর্গেশনন্দিনী/", "")
            .replace("/", " - ")
        )
        print(f"[{idx}/{len(titles)}] Fetching: {clean_label}", flush=True)

        data = api_get_with_retry({
            "action": "parse",
            "page": title,
            "prop": "text",
            "disablelimitreport": "1",
            "format": "json",
        })
        html_content = data.get("parse", {}).get("text", {}).get("*", "")
        clean_text = parse_chapter_html(html_content)

        if clean_text:
            page_url = unquote(BASE_WIKI_URL + quote(title.replace(" ", "_")))
            chapters.append({
                "chapter": clean_label,
                "title": title,
                "source_url": page_url,
                "text": clean_text,
            })
        time.sleep(1.5)

    chapters.sort(key=lambda item: chapter_sort_key(item["chapter"]))

    if chapters:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(chapters, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chapters)} sorted chapters to {CACHE_FILE}.", flush=True)

    return chapters


if __name__ == "__main__":
    fetch_all_chapters()
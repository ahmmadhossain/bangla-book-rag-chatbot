# দুর্গেশনন্দিনী — Bangla Book Knowledge Base Chatbot

A Bengali RAG chatbot for Bankim Chandra Chattopadhyay's **দুর্গেশনন্দিনী**.
It crawls the Wikisource index and chapter subpages, creates multilingual
embeddings, stores them in Chroma DB, and answers questions with chapter citations.

## Book source

- Book: [দুর্গেশনন্দিনী](https://bn.wikisource.org/wiki/দুর্গেশনন্দিনী)
- Wikisource index / table of contents:
  [নির্ঘণ্ট:দুর্গেশনন্দিনী বঙ্কিমচন্দ্র চট্টোপাধ্যায়.djvu](https://bn.wikisource.org/wiki/নির্ঘণ্ট:দুর্গেশনন্দিনী_বঙ্কিমচন্দ্র_চট্টোপাধ্যায়.djvu)
- Author: বঙ্কিমচন্দ্র চট্টোপাধ্যায়

The crawler reads the table of contents from the Wikisource index and follows
the chapter links under `দুর্গেশনন্দিনী/`. It does not intentionally stop at
the book's landing page. Check the crawler's final chapter count and inspect
`data/raw/durgeshnandini_chapters.json` after ingestion.

## Architecture

Wikisource index + chapter pages → Requests/BeautifulSoup crawler → chapter-aware
documents → overlapping chunks → multilingual sentence embeddings → Chroma →
LangChain retriever → OpenAI-compatible chat model → Streamlit interface.

## Technology choices

- **Data crawling & caching:** `requests` and `BeautifulSoup` (`crawler.py`) fetch chapters of *Durgeshnandini* through the Bengali Wikisource API with retry handling for rate limits (`HTTP 429`). Non-story elements are stripped, chapters are sorted into true reading order using Bengali volume/chapter ordinals, and the cleaned text is cached locally in `durgeshnandini_chapters.json`.
- **Embedding model:** `intfloat/multilingual-e5-base` (via `langchain-huggingface`, normalized embeddings on CPU). It is a multilingual sentence-transformer suited for Bengali semantic similarity while remaining much smaller and more practical for a student laptop than larger models such as BGE-M3. Chunks are prefixed with `passage: [<chapter>]` during indexing and queries with `query:` at retrieval time to match E5's expected format.
- **Vector database:** ChromaDB (`chromadb` + `langchain-chroma`), persisted locally at `./chroma_durgeshnandini`. It runs locally with no separate database server required while providing built-in collection management and rich metadata filtering.
- **Chunking:** `RecursiveCharacterTextSplitter` with **800 characters** and **200 characters of overlap**, using Bengali-aware separators including the danda (`। `). This preserves nearby narrative and dialogue context while reducing the risk of splitting key exchanges across chunk boundaries. Each chunk retains `chapter`, `chapter_num`, and `source_url` as metadata.
- **RAG & retrieval:** Multi-chapter two-hop hybrid retrieval combining Bengali stemmed keyword search over cached chapters with dense vector search from ChromaDB (`k=12`), followed by a second-hop expansion using key terms found in Hop 1. Results are deduplicated with a per-chapter diversity cap (at most 2 chunks per chapter, up to 9 chunks total) and sorted in chronological story order so plot and character reveals stay coherent. The prompt instructs the LLM to rely strictly on retrieved passages, synthesize multi-chapter character details, cite volume and chapter names, and state when an answer is not found.
- **LLM & model routing:** `ChatOpenAI` (`langchain-openai`) connected to the OpenAI-compatible CheaperInference API (`https://api.cheaperinference.com/v1`). Text models are automatically ordered by highest discount (starting with free models such as `ampersand-v0.1`, `beam-free`, `flint-v1.0`, `glyph-cluster`, and `nemotron-3.5-lightning-free`), with automatic failover and cooldown switching to the next available model when token quotas, rate limits, or empty responses occur.
- **Frontend:** Streamlit, configured with `.env` (`python-dotenv`) for local development and Streamlit Secrets (`st.secrets`) for Streamlit Cloud deployment.

## Requirements

- Python 3.14 (the version used in this project setup) or another version
  supported by the installed PyTorch / sentence-transformers / FAISS wheels.
- Internet access for the first crawl and model download.
- An API key for an OpenAI-compatible chat-completion provider.
- Approximately 1–2 GB free disk space is recommended for Python packages,
  model files, and the local vector index.

## Windows setup (PowerShell)

```powershell
cd path\to\bangla-book-rag-chatbot

# Create and activate a virtual environment
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, use this for the current terminal only:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Create your local environment file
Copy-Item .env.example .env
notepad .env
```

Put your provider's API key and model name in `.env`. For an OpenAI-compatible
provider, set `BASE_URL` to its documented API base URL. Do not commit
`.env` or expose the key in screenshots or the demo.

### Python version note

The setup command above uses Python 3.14. If pip reports that a package has no
compatible wheel for your Windows/Python combination, update pip and retry.
Do not change or uninstall your system Python just to create this project venv.

## Run the project

Run these commands from the project root, with the virtual environment active.

```powershell
# Step 1: crawl the index and all discovered chapter pages
python crawler.py

# Step 2: chunk the book, embed it, and create the FAISS index
python ingest.py

# Step 3: start the chatbot
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

You can also run `python ingest.py` directly on a fresh project: it automatically
runs the crawler if the raw chapter JSON file is missing.

## Test questions

The 10 test cases and expected-answer criteria are in
[`test_questions.json`](test_questions.json). They cover author,
chapter titles, chapter locations, table-of-contents facts, and one out-of-book
question to test refusal. Run all questions manually in the Streamlit UI and
compare the answer and cited chapter with the expected answer criteria.

## Important evaluation note

The chatbot's refusal is based on retrieved context and prompt instructions;
LLMs are not mathematically guaranteed never to hallucinate. For a stronger
submission, manually test the 10 questions, verify each cited chapter URL, and
record any failures. The index metadata says some page scans have not all been
proofread, so inspect the downloaded text for completeness and quality.

## GitHub

Create a new public repository, then:

```powershell
git init
git add .
git commit -m "Build Bangla book RAG chatbot"
git branch -M main
git remote add origin https://github.com/ahmmadhossain/bangla-book-rag-chatbot.git
git push -u origin main
```

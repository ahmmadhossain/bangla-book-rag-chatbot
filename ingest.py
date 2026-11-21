import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from crawler import fetch_all_chapters

CHROMA_DIR = "./chroma_durgeshnandini"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def build_vector_store():
    chapters = fetch_all_chapters()
    if not chapters:
        raise RuntimeError("No chapter text extracted from Wikisource.")

    raw_docs = [
        Document(
            page_content=item["text"],
            metadata={
                "chapter": item["chapter"],
                "chapter_num": idx,
                "source_url": item["source_url"],
            },
        )
        for idx, item in enumerate(chapters, start=1)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "। ", "?", "!", " ", ""],
    )

    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks from {len(raw_docs)} chapters.", flush=True)

    # Prefix each chunk with passage: and its chapter name for stronger contextual embeddings
    for chunk in chunks:
        ch_name = chunk.metadata["chapter"]
        chunk.page_content = f"passage: [{ch_name}]\n{chunk.page_content}"

    print(f"Loading embedding model: {EMBEDDING_MODEL}...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Resetting ChromaDB collection at {CHROMA_DIR}...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("durgeshnandini")
    except Exception:
        pass  # Collection does not exist yet

    print(f"Storing embeddings in ChromaDB at {CHROMA_DIR}...", flush=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="durgeshnandini",
    )
    print("Ingestion complete!", flush=True)


if __name__ == "__main__":
    build_vector_store()